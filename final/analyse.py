import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from final import read_images
from final.image_generation_with_metrics import initialize_image
from final.parameters.InitializationType import InitializationType


def plot_grad_ratio(df, save_path, label):
    """
    Plot gradient ratio. Takes DataFrame as input and saves the plot of gradient ratio column as a png file.
    Columns "content_grad_norm" and "style_grad_norm" columns used for calculation.
    :param df: DataFrame containing data, specifically the "content_grad_norm" and "style_grad_norm" columns used for calculation.
    :param save_path: path to save the plot
    :param label: label of the plot
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["content_grad_norm"] / df["style_grad_norm"],
        df["grad_ratio"],
        label=label
    )

    plt.xlabel("Optimization step")
    plt.ylabel("Content / style gradient ratio")
    plt.title("Effective content-style balance")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()


def plot_grad_ratio_comparison(df1, df2, save_path, df1_label="df1", df2_label="df2"):
    """
    Plot gradient ratio for two different csv files and saves the image.
    Columns "content_grad_norm" and "style_grad_norm" columns used for calculation.
    :param df1: first DataFrame containing data, specifically the "content_grad_norm" and "style_grad_norm" columns used for calculation.
    :param df2: second DataFrame containing data, specifically the "content_grad_norm" and "style_grad_norm" columns used for calculation.
    :param save_path: path to save the plot
    :param df1_label: label for df1
    :param df2_label: label for df2
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        df1["step"],
        df1["content_grad_norm"] / df1["style_grad_norm"],
        label=df1_label
    )

    plt.plot(
        df2["step"],
        df2["content_grad_norm"] / df2["style_grad_norm"],
        label=df2_label
    )

    plt.xlabel("Optimization step")
    plt.ylabel("Content / style gradient ratio")
    plt.title("Effective content-style balance")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()


def plot_loss_ratio(df, save_path, label, raw=False):
    """
    Plot loss ratio. Takes DataFrame as input and saves the plot of loss ratio or raw loss ratio as a png file.
    :param df: DataFrame containing data, specifically the "loss_ratio" column
    :param save_path: path to save the plot
    :param label: label of the plot
    :param raw: whether to plot raw loss ratio
    """
    plt.figure(figsize=(8, 5))

    if raw:
        plt.plot(
            df["step"],
            df["content_loss"] / df["style_loss"],
            label=label
        )
    else:
        plt.plot(
            df["step"],
            df["raw_content_loss"] / df["raw_style_loss"],
        )

    plt.xlabel("Optimization step")
    plt.ylabel("Content / style loss ratio")
    plt.title("Content-style loss ratio")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()


def plot_loss_ratio_comparison(df1, df2, save_path, df1_label="df1", df2_label="df2", raw=False):
    """
    Plot loss ratio for two different generates csv files and saves the image.
    :param df1: first DataFrame containing data, specifically the "loss_ratio" column
    :param df2: second DataFrame containing data, specifically the "loss_ratio" column
    :param save_path: path to save the plot
    :param df1_label: label for df1
    :param df2_label: label for df2
    :param raw: whether to plot raw loss ratio
    """
    plt.figure(figsize=(8, 5))

    if raw:
        plt.plot(
            df1["step"],
            df1["content_loss"] / df1["style_loss"],
            label=df1_label
        )
    else:
        plt.plot(
            df1["step"],
            df1["raw_content_loss"] / df1["raw_style_loss"],
            label=df1_label
        )

    if raw:
        plt.plot(
            df2["step"],
            df2["content_loss"] / df2["style_loss"],
            label=df2_label
        )
    else:
        plt.plot(
            df2["step"],
            df2["raw_content_loss"] / df2["raw_style_loss"],
            label=df2_label
        )

    plt.xlabel("Optimization step")
    plt.ylabel("Content / style gradient ratio")
    plt.title("Effective content-style balance")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()


def plot_losses(df, save_path, log_scale=True):
    """
    Plot style, content, and total loss over optimization steps and save the figure to disk.
    :param df: DataFrame containing data, specifically the "style_loss" and "content_loss" columns.
    :param save_path: path to save the plot
    :param log_scale: whether to use logarithmic scale
    """
    plt.figure(figsize=(9, 5))

    plt.plot(
        df["step"],
        df["style_loss"],
        label="Style loss"
    )

    plt.plot(
        df["step"],
        df["content_loss"],
        label="Content loss"
    )

    plt.plot(
        df["step"],
        df["content_loss"] + df["style_loss"],
        label="Total loss"
    )

    plt.xlabel("Optimization step")
    plt.ylabel("Loss")
    plt.title("Style, content, and total loss")

    if log_scale:
        plt.yscale("log")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.
    About SSIM: https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    :param img1: image 1
    :param img2: image 2
    :return: SSIM between img1 and img2
    """
    ssim = tf.image.ssim(
        img1,
        img2,
        max_val=1.0
    )
    print(ssim)
    return ssim


### Put code here to analyse some of the results. Example:
img1 = read_images.load_img("results/survey res/init_comp/exp1/skull2.png")
# img2 = read_files.load_img("img/content/still-life/Skull.jpg")
# calculate_ssim(img1[0], img2[0])

df = pd.read_csv("results/metrics of noise init/metrics.csv")
# plot_grad_ratio(df, "results/grad_ratio.png", "experiment name")
plot_losses(df, "results/losses2.png")
