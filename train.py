import random
import warnings

from argparse import ArgumentParser
from functools import partial

import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

from torchmetrics.regression import CosineSimilarity

from torch.utils.tensorboard import SummaryWriter

from esm.tokenization import EsmSequenceTokenizer
from esm.models.esmc import ESMC

from src.prot_hash.model import ProtHash
from data import SwissProt
from loss import DistillationLoss

from tqdm import tqdm


def main():
    parser = ArgumentParser(
        description="Distill a larger ESMC model into a smaller one."
    )

    parser.add_argument("--teacher_model_name", type=str, default="esmc_600m")
    parser.add_argument("--num_dataset_processes", default=1, type=int)
    parser.add_argument("--min_sequence_length", default=1, type=int)
    parser.add_argument("--max_sequence_length", default=2048, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--temperature", default=2.0, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--max_steps", default=20000, type=int)
    parser.add_argument("--embedding_dimensions", default=768, type=int)
    parser.add_argument("--q_heads", default=16, type=int)
    parser.add_argument("--kv_heads", default=4, type=int)
    parser.add_argument("--hidden_ratio", default=4, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--num_encoder_layers", default=10, type=int)
    parser.add_argument("--eval_interval", default=100, type=int)
    parser.add_argument("--checkpoint_interval", default=100, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.max_steps < 1:
        raise ValueError(f"Must train for at least 1 step, {args.max_steps} given.")

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

    dataset = SwissProt(
        tokenizer=tokenizer,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
    )

    training, testing = random_split(dataset, (0.99, 0.01))

    new_dataloader = partial(
        DataLoader,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_pad_right,
        pin_memory="cuda" in args.device,
        num_workers=args.num_dataset_processes,
    )

    train_loader = new_dataloader(training, shuffle=True)
    test_loader = new_dataloader(testing)

    print(f"Training samples: {len(training):,}")
    print(f"Testing samples: {len(testing):,}")

    teacher = ESMC.from_pretrained(args.teacher_model_name)

    for module in teacher.modules():
        for param in module.parameters():
            param.requires_grad = False

    teacher = teacher.to(args.device)

    teacher.eval()

    print("Teacher model loaded successfully")

    model_args = {
        "vocabulary_size": tokenizer.vocab_size,
        "padding_index": tokenizer.pad_token_id,
        "context_length": args.max_sequence_length,
        "embedding_dimensions": args.embedding_dimensions,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "hidden_ratio": args.hidden_ratio,
        "num_encoder_layers": args.num_encoder_layers,
        "dropout": args.dropout,
    }

    student = ProtHash(**model_args)

    student.add_adapter_head(teacher.embed.embedding_dim)

    student = student.to(args.device)

    print(f"Number of parameters: {student.num_params:,}")

    loss_function = DistillationLoss(args.temperature)

    optimizer = AdamW(student.parameters(), lr=args.learning_rate)

    step = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=False
        )

        student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        step += checkpoint["step"]

        print("Previous checkpoint resumed successfully")

    student.train()

    cosine_similarity_metric = CosineSimilarity()

    new_progress_bar = partial(
        tqdm,
        total=args.gradient_accumulation_steps,
        leave=False,
    )

    total_distillation_l2 = 0.0

    progress_bar = new_progress_bar(desc=f"Step {step:,}")

    print("Distilling ...")

    for index, x in enumerate(train_loader, start=1):
        x = x.to(args.device, non_blocking=True)

        with amp_context:
            with torch.no_grad():
                out_teacher = teacher.forward(x)

                y_teacher = out_teacher.hidden_states[-1]

            y_student = student.forward(x)

            loss = loss_function.forward(y_student, y_teacher)

            scaled_loss = loss / args.gradient_accumulation_steps

        scaled_loss.backward()

        total_distillation_l2 += loss.item()

        progress_bar.update(1)

        if index % args.gradient_accumulation_steps == 0:
            norm = clip_grad_norm_(student.parameters(), args.max_gradient_norm)

            optimizer.step()

            optimizer.zero_grad()

            progress_bar.close()

            average_distillation_l2 = (
                total_distillation_l2 / args.gradient_accumulation_steps
            )

            gradient_norm = norm.item()

            logger.add_scalar("Distillation L2", average_distillation_l2, step)
            logger.add_scalar("Gradient Norm", gradient_norm, step)

            print(
                f"Step {step:,}:",
                f"Distillation L2: {average_distillation_l2:.5f},",
                f"Gradient Norm: {gradient_norm:.4f}",
            )

            if step % args.eval_interval == 0:
                student.eval()

                for x in tqdm(test_loader, desc="Testing", leave=False):
                    x = x.to(args.device, non_blocking=True)

                    with torch.no_grad():
                        out_teacher = teacher.forward(x)
                        y_teacher = out_teacher.hidden_states[-1][:, 0, :]

                        y_student = student.embed(x)

                        cosine_similarity_metric.update(y_student, y_teacher)

                average_cosine_similarity = cosine_similarity_metric.compute()

                logger.add_scalar("Cosine Similarity", average_cosine_similarity, step)

                print(
                    f"Step {step}: Cosine Similarity: {average_cosine_similarity:.5f}"
                )

                cosine_similarity_metric.reset()

                student.train()

            if step % args.checkpoint_interval == 0:
                checkpoint = {
                    "step": step,
                    "model_args": model_args,
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                torch.save(checkpoint, args.checkpoint_path)

                print("Checkpoint saved")

            if step >= args.max_steps:
                break

            step += 1

            total_distillation_l2 = 0.0

            progress_bar = new_progress_bar(desc=f"Step {step:,}")

    print("Done!")


if __name__ == "__main__":
    main()
