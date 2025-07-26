# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:02:23 2025

@author: usouu
"""

import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# %% Visualization
def draw_heatmap_1d(data, yticklabels=None, xticklabels=None, figsize=(2, 10)):
    """
    Plots a heatmap for an Nx1 array (vertical orientation).

    Parameters:
        data (numpy.ndarray): Nx1 array for visualization.
        yticklabels (list, optional): Labels for the y-axis. If None, indices will be used.
    """
    if yticklabels is None:
        yticklabels = list(range(data.shape[0]))  # Automatically generate indices as labels
    if xticklabels is None:
        xticklabels = list(range(data.shape[1]))  # Automatically generate indices as labels
    
    if len(data.shape) == 1:
        data = np.reshape(data, (-1, 1))
    
    data = np.array(data, dtype=float)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        data, 
        cmap='Blues',
        cbar=False,
        annot=False,
        linewidths=0.5, 
        yticklabels=yticklabels,
        xticklabels=xticklabels
    )
    # plt.title("Vertical Heatmap of Nx1 Array")
    plt.show()

def draw_joint_heatmap_1d(data_dict):
    heatmap_labels = []
    heatmap_data = []

    for label, data in data_dict.items():
        heatmap_labels.append(label)
        heatmap_data.append(data)

    heatmap_data = np.vstack(heatmap_data) 
    heatmap_labels = np.array(heatmap_labels)
        
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='viridis', cbar=True, xticklabels=False, yticklabels=heatmap_labels, linewidths=0.5, linecolor='gray')
    plt.title("Heatmap of cw_target and All cw_fitting Vectors")
    plt.xlabel("Channel Index")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()

def draw_projection(sample_projection, title=None, xticklabels=None, yticklabels=None):
    """
    Visualizes data projections (common for both datasets).
    
    Parameters:
        sample_projection (np.ndarray): 2D or 3D matrix to visualize.
        title (str): Optional plot title.
        xticklabels (list): Optional list of x-axis labels.
        yticklabels (list): Optional list of y-axis labels.
    """
    if title is None:
        title = "2D Matrix Visualization"
    
    def apply_axis_labels(ax, xticks, yticks):
        if xticks is not None:
            ax.set_xticks(range(len(xticks)))
            ax.set_xticklabels(xticks, rotation=90)
        if yticks is not None:
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks)

    if sample_projection.ndim == 2:
        fig, ax = plt.subplots()
        im = ax.imshow(sample_projection, cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        apply_axis_labels(ax, xticklabels, yticklabels)
        plt.tight_layout()
        plt.show()

    elif sample_projection.ndim == 3 and sample_projection.shape[0] <= 100:
        for i in range(sample_projection.shape[0]):
            fig, ax = plt.subplots()
            im = ax.imshow(sample_projection[i], cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Channel {i + 1} Visualization")
            apply_axis_labels(ax, xticklabels, yticklabels)
            plt.tight_layout()
            plt.show()

    else:
        raise ValueError(f"The dimension of sample matrix for drawing is wrong, shape of sample: {sample_projection.shape}")

# %% End Program Actions
import time
import threading
def shutdown_with_countdown(countdown_seconds=30):
    """
    Initiates a shutdown countdown, allowing the user to cancel shutdown within the given time.

    Args:
        countdown_seconds (int): The number of seconds to wait before shutting down.
    """
    def cancel_shutdown():
        nonlocal shutdown_flag
        user_input = input("\nPress 'c' and Enter to cancel shutdown: ").strip().lower()
        if user_input == 'c':
            shutdown_flag = False
            print("Shutdown cancelled.")

    # Flag to determine whether to proceed with shutdown
    shutdown_flag = True

    # Start a thread to listen for user input
    input_thread = threading.Thread(target=cancel_shutdown, daemon=True)
    input_thread.start()

    # Countdown timer
    print(f"Shutdown scheduled in {countdown_seconds} seconds. Press 'c' to cancel.")
    for i in range(countdown_seconds, 0, -1):
        print(f"Time remaining: {i} seconds", end="\r")
        time.sleep(1)

    # Check the flag after countdown
    if shutdown_flag:
        print("\nShutdown proceeding...")
        os.system("shutdown /s /t 1")  # Execute shutdown command
    else:
        print("\nShutdown aborted.")

def end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120):
    """
    Performs actions at the end of the program, such as playing a sound or shutting down the system.

    Args:
        play_sound (bool): If True, plays a notification sound.
        shutdown (bool): If True, initiates shutdown with a countdown.
        countdown_seconds (int): Countdown time for shutdown confirmation.
    """
    if play_sound:
        try:
            import winsound
            print("Playing notification sound...")
            winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
        except ImportError:
            print("winsound module not available. Skipping sound playback.")

    if shutdown:
        shutdown_with_countdown(countdown_seconds)

# %% Example Usage
# if __name__ == '__main__':
