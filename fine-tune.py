import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.backends.mps import is_available as mps_is_available
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

from torchmetrics.classification import BinaryPrecision, BinaryRecall

from torch.utils.tensorboard import SummaryWriter

from esm.tokenization import EsmSequenceTokenizer

from model import EsmcGoTermClassifier
from data import AmiGOBoost

from tqdm import tqdm


AVAILABLE_BASE_MODELS = EsmcGoTermClassifier.ESM_PRETRAINED_CONFIGS.keys()


def main():
    parser = ArgumentParser(
        description="Fine-tune an ESMC model for gene ontology (GO) term prediction."
    )

    parser.add_argument(
        "--base_model",
        default="esmc_300m",
        choices=AVAILABLE_BASE_MODELS,
    )
    parser.add_argument(
        "--dataset_subset", default="all", choices=AmiGOBoost.AVAILABLE_SUBSETS
    )
    parser.add_argument("--num_dataset_processes", default=1, type=int)
    parser.add_argument("--min_sequence_length", default=1, type=int)
    parser.add_argument("--max_sequence_length", default=2048, type=int)
    parser.add_argument("--unfreeze_last_k_layers", default=7, type=int)
    parser.add_argument("--add_lora", action="store_true")
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--num_epochs", default=40, type=int)
    parser.add_argument("--classifier_hidden_ratio", default=1, type=int)
    parser.add_argument("--use_flash_attention", default=True, type=bool)
    parser.add_argument("--eval_interval", default=2, type=int)
    parser.add_argument("--checkpoint_interval", default=2, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.num_epochs < 1:
        raise ValueError(f"Must train for at least 1 epoch, {args.num_epochs} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if "mps" in args.device and not mps_is_available():
        raise RuntimeError("MPS is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if ("cuda" in args.device and is_bf16_supported()) or "mps" in args.device
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    tokenizer = EsmSequenceTokenizer()

    new_dataset = partial(
        AmiGOBoost,
        subset=args.dataset_subset,
        tokenizer=tokenizer,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
    )

    training = new_dataset(split="train")
    testing = new_dataset(split="test")

    new_dataloader = partial(
        DataLoader,
        batch_size=args.batch_size,
        collate_fn=training.collate_pad_right,
        pin_memory="cuda" in args.device,
        num_workers=args.num_dataset_processes,
    )

    train_loader = new_dataloader(training, shuffle=True)
    test_loader = new_dataloader(testing)

    model_args = {
        "model_name": args.base_model,
        "classifier_hidden_ratio": args.classifier_hidden_ratio,
        "id2label": training.label_indices_to_terms,
        "use_flash_attention": args.use_flash_attention,
    }

    model = EsmcGoTermClassifier.from_esm_pretrained(**model_args)

    print(f"Number of parameters: {model.num_params:,}")

    model.freeze_base()

    model.unfreeze_last_k_encoder_layers(args.unfreeze_last_k_layers)

    if args.add_lora:
        k = model.num_encoder_layers - args.unfreeze_last_k_layers

        model.add_lora_to_first_k_encoder_layers(k, args.lora_rank)

    print(f"Number of trainable parameters: {model.num_trainable_parameters:,}")

    model = model.to(args.device)

    loss_function = BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    precision_metric = BinaryPrecision().to(args.device)
    recall_metric = BinaryRecall().to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=False
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    model.train()

    logger = SummaryWriter(args.run_dir_path)

    print("Fine-tuning ...")

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with amp_context:
                y_pred = model.forward(x)

                loss = loss_function(y_pred, y)

                scaled_loss = loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            total_cross_entropy += loss.item()
            total_batches += 1

            if step % args.gradient_accumulation_steps == 0:
                norm = clip_grad_norm_(model.parameters(), args.max_gradient_norm)

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                total_gradient_norm += norm.item()
                total_steps += 1

        average_cross_entropy = total_cross_entropy / total_batches
        average_gradient_norm = total_gradient_norm / total_steps

        logger.add_scalar("Cross Entropy", average_cross_entropy, epoch)
        logger.add_scalar("Gradient Norm", average_gradient_norm, epoch)

        print(
            f"Epoch {epoch}:",
            f"Cross Entropy: {average_cross_entropy:.5f},",
            f"Gradient Norm: {average_gradient_norm:.5f}",
        )

        if epoch % args.eval_interval == 0:
            model.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                with torch.no_grad(), amp_context:
                    y_pred = model.forward(x)

                    y_prob = torch.sigmoid(y_pred)

                precision_metric.update(y_prob, y)
                recall_metric.update(y_prob, y)

            precision = precision_metric.compute()
            recall = recall_metric.compute()

            f1_score = (2 * precision * recall) / (precision + recall)

            logger.add_scalar("F1 Score", f1_score, epoch)
            logger.add_scalar("Precision", precision, epoch)
            logger.add_scalar("Recall", recall, epoch)

            print(
                f"F1: {f1_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}"
            )

            precision_metric.reset()
            recall_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_args": model_args,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    print("Done!")


if __name__ == "__main__":
    main()
