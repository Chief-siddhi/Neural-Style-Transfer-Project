import argparse
import itertools
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# Handles both folder structures (in root or inside utils/ directory)
try:
    from utils.utils import ImageFolderDataset, get_transform, adaptive_instance_normalization, calc_mean_std
    from utils.models import VGGEncoder, Decoder
except ModuleNotFoundError:
    from utils import ImageFolderDataset, get_transform, adaptive_instance_normalization, calc_mean_std
    from models import VGGEncoder, Decoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, default='D:/Study Folder/prime/NeuralStyleTransferProject/content_data')
    parser.add_argument('--style_dir', type=str, default='D:/Study Folder/prime/NeuralStyleTransferProject/style_data')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--experiment', type=str, default='experiment1')
    
    parser.add_argument('--content_size', type=int, default=512)
    parser.add_argument('--style_size', type=int, default=512)
    parser.add_argument('--final_size', type=int, default=256)
    parser.add_argument('--crop', action='store_true', default=True)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000, help='Total training iterations')
    
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--style_weight', type=float, default=3.0)

    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--decoder_path', type=str, default=None)
    parser.add_argument('--optimizer_path', type=str, default=None)
    return parser.parse_args()


def adjust_learning_rate(optimizer, iteration_count, lr, lr_decay):
    new_lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}:{value}\n')

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    # Windows-safe num_workers=0 avoids multi-threading pickling issues
    content_dataloader = DataLoader(
        content_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True, num_workers=0
    )
    style_dataloader = DataLoader(
        style_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True, num_workers=0
    )

    print('Number of batches in content dataset:', len(content_dataloader))
    print('Number of batches in style dataset:', len(style_dataloader))

    content_iter = iter(itertools.cycle(content_dataloader))
    style_iter = iter(itertools.cycle(style_dataloader))

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    encoder.eval()
    decoder.train()

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    if args.resume and args.decoder_path:
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        if args.optimizer_path:
            optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))
        print(f"Resumed weights from {args.decoder_path}")

    mse_loss = nn.MSELoss()
    progress_bar = tqdm(range(args.max_iter), initial=0, total=args.max_iter)

    for step in progress_bar:
        adjust_learning_rate(optimizer, step, args.lr, args.lr_decay)

        content_batch = next(content_iter).to(device)
        style_batch = next(style_iter).to(device)

        with torch.no_grad():
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)
            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

        g = decoder(t)
        g = torch.clamp(g, 0.0, 1.0)

        g_feats = encoder(g)

        loss_c = mse_loss(g_feats[-1], t) * args.content_weight

        loss_s = 0.0
        for g_f, s_f in zip(g_feats, s_feats):
            g_mean, g_std = calc_mean_std(g_f)
            s_mean, s_std = calc_mean_std(s_f)
            loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)

        loss_s = loss_s * args.style_weight
        loss = loss_c + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % args.log_interval == 0:
            progress_bar.set_description(
                f"Iter {step+1}/{args.max_iter} | Loss: {loss.item():.4f} | "
                f"C_Loss: {loss_c.item():.4f} | S_Loss: {loss_s.item():.4f}"
            )

        if (step + 1) % args.save_interval == 0 or (step + 1) == args.max_iter:
            torch.save(decoder.state_dict(), save_dir / f'decoder_iter_{step+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_iter_{step+1}.pth')

            with torch.no_grad():
                sample_vis = torch.cat([content_batch[:4], style_batch[:4], g[:4]], dim=0)
                save_image(sample_vis, save_dir / f'output_iter_{step+1}.png', nrow=4)


if __name__ == '__main__':
    main()
       