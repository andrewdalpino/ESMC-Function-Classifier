import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

from torchmetrics.classification import BinaryPrecision, BinaryRecall

from torch.utils.tensorboard import SummaryWriter

from esm.tokenization import EsmSequenceTokenizer

import obonet

from src.esmc_function_classifier.model import EsmcGoTermClassifier
from data import AmiGOBoost
from loss import DistillationLoss

from tqdm import tqdm


AVAILABLE_BASE_MODELS = EsmcGoTermClassifier.ESM_PRETRAINED_CONFIGS.keys()


def main():
    parser = ArgumentParser(
        description="Distill a larger fine-tuned model into a smaller one."
    )

    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset_subset", default="all", choices=AmiGOBoost.AVAILABLE_SUBSETS
    )
    parser.add_argument("--num_dataset_processes", default=1, type=int)
    parser.add_argument("--go_db_path", default="./dataset/go-basic.obo", type=str)
    parser.add_argument("--min_sequence_length", default=1, type=int)
    parser.add_argument("--max_sequence_length", default=2048, type=int)
    parser.add_argument("--quantization_aware_training", action="store_true")
    parser.add_argument("--quant_group_size", default=192, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--teacher_alpha", default=0.5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--embedding_dimensions", default=768, type=int)
    parser.add_argument("--num_heads", default=12, type=int)
    parser.add_argument("--num_encoder_layers", default=10, type=int)
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
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    tokenizer = EsmSequenceTokenizer()

    graph = obonet.read_obo(args.go_db_path)

    new_dataset = partial(
        AmiGOBoost,
        subset=args.dataset_subset,
        graph=graph,
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

    print(f"Training samples: {len(training.dataset):,}")
    print(f"Testing samples: {len(testing.dataset):,}")

    checkpoint = torch.load(
        args.teacher_checkpoint, map_location=args.device, weights_only=False
    )

    teacher_args = checkpoint["model_args"]

    teacher = EsmcGoTermClassifier.from_esm_pretrained(**teacher_args)

    teacher.load_state_dict(checkpoint["model"])

    teacher.remove_fake_quantized_tensors()

    print("Teacher model loaded successfully")

    student_args = {
        "embedding_dimensions": args.embedding_dimensions,
        "num_heads": args.num_heads,
        "num_encoder_layers": args.num_encoder_layers,
        "classifier_hidden_ratio": args.classifier_hidden_ratio,
        "id2label": training.label_indices_to_go_ids,
        "use_flash_attention": args.use_flash_attention,
    }

    student = EsmcGoTermClassifier(**student_args)

    if args.quantization_aware_training:
        student.add_fake_quantized_tensors(args.quant_group_size)

    print(f"Number of parameters: {student.num_params:,}")

    student = student.to(args.device)

    loss_function = DistillationLoss(args.teacher_alpha)

    optimizer = AdamW(student.parameters(), lr=args.learning_rate)

    precision_metric = BinaryPrecision().to(args.device)
    recall_metric = BinaryRecall().to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=False
        )

        student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    student.train()

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
                y_student = student.forward(x)

                with torch.no_grad():
                    y_teacher = teacher.forward(x)

                loss = loss_function(y_student, y_teacher, y)

                scaled_loss = loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            total_cross_entropy += loss.item()
            total_batches += 1

            if step % args.gradient_accumulation_steps == 0:
                norm = clip_grad_norm_(student.parameters(), args.max_gradient_norm)

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
            student.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                with torch.no_grad(), amp_context:
                    y_pred = student.forward(x)

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

            student.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "student_args": student_args,
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    print("Done!")


if __name__ == "__main__":
    main()
