#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 21:54:29 2025

@author: junaidrehman
"""

import sys
import numpy as np
import random
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import networkx as nx
import time
import math
import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
                             QTextEdit, QProgressBar, QFileDialog, QMessageBox, QSplitter,
                             QFormLayout, QCheckBox, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import pyqtgraph as pg

# Import your simulation code (make sure Qsim_v2 is available)
try:
    from Qsim_v4 import *
except ImportError:
    # Fallback for demonstration purposes
    class FallbackQsim:
        @staticmethod
        def simulate_source_destination(args):
            G, sn, dest, modes, graph_name, eps = args
            # Mock simulation for demonstration
            results = []
            for mode in modes:
                fidelity = random.uniform(0.6, 0.99)
                rate = random.uniform(0.7, 1.0)
                path_len = random.randint(2, 10)
                ext_path_len = random.randint(3, 12)
                results.append((mode, fidelity, rate, sn, dest, path_len, ext_path_len))
            return results

        @staticmethod
        def gnp_random_connected_graph(n, p):
            # Create a random connected graph for demonstration
            while True:
                G = nx.erdos_renyi_graph(n, p)
                if nx.is_connected(G):
                    return G


    simulate_source_destination = FallbackQsim.simulate_source_destination
    gnp_random_connected_graph = FallbackQsim.gnp_random_connected_graph


class SimulationThread(QThread):
    """Thread to run the simulation without freezing the GUI"""
    progress_updated = pyqtSignal(int)
    simulation_finished = pyqtSignal(list, list, list, list, list, list, str)
    log_message = pyqtSignal(str)
    intermediate_results = pyqtSignal(list, list, list, list, list, list)  # For dynamic updates
    graph_updated = pyqtSignal(object)  # For graph visualization

    def __init__(self, nodes, probability, N, eps, modes, fidelity_range, rate_range, pur_round_range,
                 swap_success_range, save_data):
        super().__init__()
        self.nodes = nodes
        self.probability = probability
        self.N = N
        self.eps = eps
        self.modes = modes
        self.fidelity_range = fidelity_range
        self.rate_range = rate_range
        self.pur_round_range = pur_round_range
        self.swap_success_range = swap_success_range
        self.save_data = save_data
        self.is_running = True
        self.G = None
        self.num_processes = max(1, multiprocessing.cpu_count() - 2)

    def run(self):
        try:
            # Generate the random connected graph
            self.log_message.emit("Generating graph...")
            np.random.seed(10)
            random.seed(10)
            self.G = gnp_random_connected_graph(self.nodes, self.probability)

            # Assign properties to edges with user-defined ranges
            for u, v in self.G.edges():
                self.G[u][v]['fidelity'] = np.random.uniform(low=self.fidelity_range[0], high=self.fidelity_range[1])
                self.G[u][v]['rate'] = np.random.uniform(low=self.rate_range[0], high=self.rate_range[1])
                self.G[u][v]['pur_round'] = np.random.randint(low=self.pur_round_range[0], high=self.pur_round_range[1])

            # Assign properties to nodes with user-defined swap success probability
            for node in self.G.nodes():
                self.G.nodes[node]['swap_success'] = np.random.uniform(low=self.swap_success_range[0],
                                                                       high=self.swap_success_range[1])

            # Emit graph for visualization
            self.graph_updated.emit(self.G)

            graph_name = f"Random-{self.nodes}-probability-{self.probability}"

            # Initialize data storage for results
            mode_1_fid, mode_1_rate = [], []
            mode_2_fid, mode_2_rate = [], []
            mode_3_fid, mode_3_rate = [], []

            # Generate the random source-destination pairs for parallel processing
            pairs = [(np.random.randint(low=0, high=self.nodes),
                      np.random.randint(low=0, high=self.nodes)) for _ in range(self.N)]
            pairs = [pair for pair in pairs if pair[0] != pair[1]]  # Remove pairs with same source and destination

            # Define the pool for parallel execution
            results = []
            with Pool(processes=self.num_processes) as pool:
                # Process pairs in chunks to update progress
                #chunk_size = max(1, len(pairs) // 50)  # More chunks for more frequent updates
                chunk_size = 2*self.num_processes
                for i in range(0, len(pairs), chunk_size):
                    if not self.is_running:
                        break

                    chunk = pairs[i:i + chunk_size]
                    chunk_results = list(pool.imap(simulate_source_destination,
                                                   [(self.G, sn, dest, self.modes, graph_name, self.eps) for sn, dest in
                                                    chunk]))
                    results.extend(chunk_results)

                    # Collect intermediate results for dynamic updates
                    for result in chunk_results:
                        for mode, best_fid, best_rate, sn, dest, path_len, ext_path_len in result:
                            if best_fid is not None:
                                if mode == 1:
                                    mode_1_fid.append(best_fid)
                                    mode_1_rate.append(best_rate)
                                elif mode == 2:
                                    mode_2_fid.append(best_fid)
                                    mode_2_rate.append(best_rate)
                                elif mode == 3:
                                    mode_3_fid.append(best_fid)
                                    mode_3_rate.append(best_rate)

                    # Emit intermediate results for dynamic plotting
                    self.intermediate_results.emit(
                        mode_1_fid.copy(), mode_1_rate.copy(),
                        mode_2_fid.copy(), mode_2_rate.copy(),
                        mode_3_fid.copy(), mode_3_rate.copy()
                    )

                    # Update progress
                    progress = int((i + len(chunk)) / len(pairs) * 100)
                    self.progress_updated.emit(progress)

                    # Small delay to allow GUI updates
                    time.sleep(0.1)
                    #self.reset_button.setEnabled(True)

            if not self.is_running:
                return

            # Find the minimum length among all arrays
            kk = min(len(arr) for arr in [mode_1_fid, mode_1_rate, mode_2_fid, mode_2_rate, mode_3_fid, mode_3_rate])

            # Truncate each array to the minimum length
            mode_1_fid, mode_1_rate, mode_2_fid, mode_2_rate, mode_3_fid, mode_3_rate = \
                [arr[:kk] for arr in [mode_1_fid, mode_1_rate, mode_2_fid, mode_2_rate, mode_3_fid, mode_3_rate]]

            # Generate filename
            fname_CDF = f"nodes_{self.nodes}_N_{self.N}_Episodes_{self.eps}_Probability_{int(1000 * self.probability)}_CDF.dat"

            # Save data if checkbox is checked
            if self.save_data:
                try:
                    # Create data directory if it doesn't exist
                    os.makedirs("data_clean_v2", exist_ok=True)

                    # Prepare data for saving
                    m1fx = np.sort(mode_1_fid)
                    m1fy = np.arange(len(m1fx)) / float(len(m1fx))
                    m2fx = np.sort(mode_2_fid)
                    m2fy = np.arange(len(m2fx)) / float(len(m2fx))
                    m3fx = np.sort(mode_3_fid)
                    m3fy = np.arange(len(m3fx)) / float(len(m3fx))

                    m1rx = np.sort(mode_1_rate)
                    m1ry = np.arange(len(m1rx)) / float(len(m1rx))
                    m2rx = np.sort(mode_2_rate)
                    m2ry = np.arange(len(m2rx)) / float(len(m2rx))
                    m3rx = np.sort(mode_3_rate)
                    m3ry = np.arange(len(m3rx)) / float(len(m3rx))

                    data = np.array([m1rx, m1ry, m2rx, m2ry, m3rx, m3ry,
                                     m1fx, m1fy, m2fx, m2fy, m3fx, m3fy]).T

                    np.savetxt(f"data_clean_v2/{fname_CDF}", data, delimiter="\t",
                               header="X1\tY1\tX2\tY2\tX3\tY3\tX4\tY4\tX5\tY5\tX6\tY6", comments="")

                    self.log_message.emit(f"Data saved to data_clean_v2/{fname_CDF}")
                except Exception as e:
                    self.log_message.emit(f"Error saving data: {str(e)}")

            # Emit results
            self.simulation_finished.emit(
                mode_1_fid, mode_1_rate,
                mode_2_fid, mode_2_rate,
                mode_3_fid, mode_3_rate,
                fname_CDF
            )

        except Exception as e:
            self.log_message.emit(f"Error: {str(e)}")

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Network Simulator")
        self.setGeometry(100, 100, 1400, 800)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for parameters and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(500)

        # Parameters group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QFormLayout(params_group)

        # Set form layout to left align labels
        params_layout.setFormAlignment(Qt.AlignLeft)
        params_layout.setLabelAlignment(Qt.AlignLeft)

        # Store input widgets for easy access
        self.input_widgets = []

        # Find the longest label to set consistent field sizes
        labels = [
            "Number of Nodes:",
            "Connection Probability:",
            "Number of Iterations:",
            "Episodes per Agent:",
            "Fidelity Range (Low-High):",
            "Rate Range (Low-High):",
            "Purification Rounds (Low-High):",
            "Swap Success Probability (Low-High):"
        ]
        longest_label = max(labels, key=len)

        # Create a label to measure the maximum width
        temp_label = QLabel(longest_label)
        temp_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        temp_label.adjustSize()
        max_label_width = temp_label.width() + 20  # Add some padding

        # Nodes parameter
        nodes_label = QLabel("Number of Nodes:")
        nodes_label.setAlignment(Qt.AlignLeft)
        nodes_label.setFixedWidth(max_label_width)
        self.nodes_spin = QSpinBox()
        self.nodes_spin.setRange(5, 100)
        self.nodes_spin.setValue(20)
        params_layout.addRow(nodes_label, self.nodes_spin)
        self.input_widgets.append(self.nodes_spin)

        # Probability parameter
        prob_label = QLabel("Connection Probability:")
        prob_label.setAlignment(Qt.AlignLeft)
        prob_label.setFixedWidth(max_label_width)
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.001, 1.0)
        self.prob_spin.setValue(0.001)
        self.prob_spin.setSingleStep(0.001)
        self.prob_spin.setDecimals(3)
        params_layout.addRow(prob_label, self.prob_spin)
        self.input_widgets.append(self.prob_spin)

        # Iterations parameter
        iter_label = QLabel("Number of Iterations:")
        iter_label.setAlignment(Qt.AlignLeft)
        iter_label.setFixedWidth(max_label_width)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 10000)
        self.iter_spin.setValue(500)
        params_layout.addRow(iter_label, self.iter_spin)
        self.input_widgets.append(self.iter_spin)

        # Episodes parameter
        ep_label = QLabel("Episodes per Agent:")
        ep_label.setAlignment(Qt.AlignLeft)
        ep_label.setFixedWidth(max_label_width)
        self.ep_spin = QSpinBox()
        self.ep_spin.setRange(100, 100000)
        self.ep_spin.setValue(5000)
        params_layout.addRow(ep_label, self.ep_spin)
        self.input_widgets.append(self.ep_spin)

        # Fidelity range
        fid_label = QLabel("Fidelity Range (Low-High):")
        fid_label.setAlignment(Qt.AlignLeft)
        fid_label.setFixedWidth(max_label_width)
        fidelity_layout = QHBoxLayout()
        self.fid_low_spin = QDoubleSpinBox()
        self.fid_low_spin.setRange(0.0, 1.0)
        self.fid_low_spin.setValue(0.65)
        self.fid_low_spin.setSingleStep(0.01)
        self.fid_low_spin.setDecimals(3)
        fidelity_layout.addWidget(self.fid_low_spin)
        self.input_widgets.append(self.fid_low_spin)

        self.fid_high_spin = QDoubleSpinBox()
        self.fid_high_spin.setRange(0.0, 1.0)
        self.fid_high_spin.setValue(0.99)
        self.fid_high_spin.setSingleStep(0.01)
        self.fid_high_spin.setDecimals(3)
        fidelity_layout.addWidget(self.fid_high_spin)
        self.input_widgets.append(self.fid_high_spin)

        params_layout.addRow(fid_label, fidelity_layout)

        # Rate range
        rate_label = QLabel("Rate Range (Low-High):")
        rate_label.setAlignment(Qt.AlignLeft)
        rate_label.setFixedWidth(max_label_width)
        rate_layout = QHBoxLayout()
        self.rate_low_spin = QDoubleSpinBox()
        self.rate_low_spin.setRange(0.0, 1.0)
        self.rate_low_spin.setValue(0.75)
        self.rate_low_spin.setSingleStep(0.01)
        self.rate_low_spin.setDecimals(3)
        rate_layout.addWidget(self.rate_low_spin)
        self.input_widgets.append(self.rate_low_spin)

        self.rate_high_spin = QDoubleSpinBox()
        self.rate_high_spin.setRange(0.0, 1.0)
        self.rate_high_spin.setValue(1.0)
        self.rate_high_spin.setSingleStep(0.01)
        self.rate_high_spin.setDecimals(3)
        rate_layout.addWidget(self.rate_high_spin)
        self.input_widgets.append(self.rate_high_spin)

        params_layout.addRow(rate_label, rate_layout)

        # Purification rounds range
        pur_label = QLabel("Purification Rounds (Low-High):")
        pur_label.setAlignment(Qt.AlignLeft)
        pur_label.setFixedWidth(max_label_width)
        pur_layout = QHBoxLayout()
        self.pur_low_spin = QSpinBox()
        self.pur_low_spin.setRange(0, 20)
        self.pur_low_spin.setValue(0)
        pur_layout.addWidget(self.pur_low_spin)
        self.input_widgets.append(self.pur_low_spin)

        self.pur_high_spin = QSpinBox()
        self.pur_high_spin.setRange(2, 20)
        self.pur_high_spin.setValue(3)
        pur_layout.addWidget(self.pur_high_spin)
        self.input_widgets.append(self.pur_high_spin)

        params_layout.addRow(pur_label, pur_layout)

        # Swap success probability range
        swap_label = QLabel("Swap Success Probability (Low-High):")
        swap_label.setAlignment(Qt.AlignLeft)
        swap_label.setFixedWidth(max_label_width)
        swap_layout = QHBoxLayout()
        self.swap_low_spin = QDoubleSpinBox()
        self.swap_low_spin.setRange(0.0, 1.0)
        self.swap_low_spin.setValue(0.23)
        self.swap_low_spin.setSingleStep(0.01)
        self.swap_low_spin.setDecimals(3)
        swap_layout.addWidget(self.swap_low_spin)
        self.input_widgets.append(self.swap_low_spin)

        self.swap_high_spin = QDoubleSpinBox()
        self.swap_high_spin.setRange(0.0, 1.0)
        self.swap_high_spin.setValue(0.8)
        self.swap_high_spin.setSingleStep(0.01)
        self.swap_high_spin.setDecimals(3)
        swap_layout.addWidget(self.swap_high_spin)
        self.input_widgets.append(self.swap_high_spin)

        params_layout.addRow(swap_label, swap_layout)

        left_layout.addWidget(params_group)

        # Reset Button
        self.reset_button = QPushButton("Reset Parameters")
        self.reset_button.clicked.connect(self.reset_simulation)
        left_layout.addWidget(self.reset_button)

        # Save data checkbox
        self.save_checkbox = QCheckBox("Save data to disk automatically")
        self.save_checkbox.setChecked(True)
        left_layout.addWidget(self.save_checkbox)

        # Buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        self.save_button = QPushButton("Save Results Manually")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        left_layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        left_layout.addWidget(self.log_output)

        # Right panel for plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create plot widgets
        plot_layout = QHBoxLayout()

        # Fidelity CDF plot
        self.fidelity_plot = pg.PlotWidget(title="Fidelity CDF")
        self.fidelity_plot.setLabel('left', 'CDF')
        self.fidelity_plot.setLabel('bottom', 'Fidelity')
        self.fidelity_plot.showGrid(True, True)
        self.fidelity_plot.setXRange(0.4, 1)
        self.fidelity_plot.setYRange(0, 1)

        # Rate CDF plot (using log10 scale)
        self.rate_plot = pg.PlotWidget(title="Rate CDF (log10 scale)")
        self.rate_plot.setLabel('left', 'CDF')
        self.rate_plot.setLabel('bottom', 'log10(Rate)')
        self.rate_plot.showGrid(True, True)
        self.rate_plot.setXRange(-4, 0)  # log10(0.001) to log10(1)

        plot_layout.addWidget(self.fidelity_plot)
        plot_layout.addWidget(self.rate_plot)

        right_layout.addLayout(plot_layout)

        # Scatter plot (using log10 for rate)
        self.scatter_plot = pg.PlotWidget(title="log10(Rate) vs Fidelity Scatter Plot")
        self.scatter_plot.setLabel('left', 'log10(Rate)')
        self.scatter_plot.setLabel('bottom', 'Fidelity')
        self.scatter_plot.showGrid(True, True)
        self.scatter_plot.setXRange(0.4, 1)
        self.scatter_plot.setYRange(-4, 0)  # log10(0.001) to log10(1)
        right_layout.addWidget(self.scatter_plot)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Initialize simulation thread
        self.simulation_thread = None

        # Store results
        self.results = None

        # Store scatter plot items
        self.scatter_items = {}

        # Log initial message
        self.log_message("Application started. Configure parameters and click 'Run Simulation'.")

    def reset_simulation(self):
        # reset all parameters
        self.nodes_spin.setValue(20)
        self.prob_spin.setValue(0.001)
        self.iter_spin.setValue(500)
        self.ep_spin.setValue(5000)
        self.fid_low_spin.setValue(0.65)
        self.fid_high_spin.setValue(0.99)
        self.rate_low_spin.setValue(0.75)
        self.rate_high_spin.setValue(1.0)
        self.pur_low_spin.setValue(0)
        self.pur_high_spin.setValue(3)
        self.swap_low_spin.setValue(0.23)
        self.swap_high_spin.setValue(0.8)

    def set_inputs_enabled(self, enabled):
        """Enable or disable all input widgets"""
        for widget in self.input_widgets:
            widget.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        self.save_checkbox.setEnabled(enabled)

    def log_message(self, message):
        """Add a message to the log output"""
        self.log_output.append(f"{message}")

    def run_simulation(self):
        """Start the simulation in a separate thread"""
        # Get parameters from UI
        nodes = self.nodes_spin.value()
        probability = self.prob_spin.value()
        N = self.iter_spin.value()
        eps = self.ep_spin.value()
        modes = [1, 2, 3]
        fidelity_range = (self.fid_low_spin.value(), self.fid_high_spin.value())
        rate_range = (self.rate_low_spin.value(), self.rate_high_spin.value())
        pur_round_range = (self.pur_low_spin.value(), self.pur_high_spin.value())
        swap_success_range = (self.swap_low_spin.value(), self.swap_high_spin.value())
        save_data = self.save_checkbox.isChecked()

        # Update UI - disable inputs, enable stop button
        self.set_inputs_enabled(False)
        self.stop_button.setEnabled(True)
        self.reset_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setEnabled(True)
        self.progress_bar.setValue(0)

        # Clear previous plots
        self.fidelity_plot.clear()
        self.rate_plot.clear()
        self.scatter_plot.clear()
        self.scatter_items = {}

        # Create and start simulation thread
        self.simulation_thread = SimulationThread(
            nodes, probability, N, eps, modes,
            fidelity_range, rate_range, pur_round_range,
            swap_success_range, save_data
        )
        self.simulation_thread.progress_updated.connect(self.update_progress)
        self.simulation_thread.simulation_finished.connect(self.simulation_done)
        self.simulation_thread.log_message.connect(self.log_message)
        self.simulation_thread.intermediate_results.connect(self.update_plots)
        self.simulation_thread.graph_updated.connect(self.visualize_graph)
        self.simulation_thread.start()

        self.log_message(
            f"Starting simulation with {nodes} nodes, probability {probability}, {N} iterations, {eps} episodes")

    def stop_simulation(self):
        """Stop the running simulation"""
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.stop()
            self.simulation_thread.wait()
            self.log_message("Simulation stopped by user")
            self.progress_bar.setEnabled(False)

        # Re-enable inputs
        self.set_inputs_enabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)

    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)

    def update_plots(self, m1f, m1r, m2f, m2r, m3f, m3r):
        """Update plots with intermediate results"""
        # Update CDF plots
        self.update_cdf_plots(m1f, m1r, m2f, m2r, m3f, m3r)

        # Update scatter plot
        self.update_scatter_plot(m1f, m1r, m2f, m2r, m3f, m3r)

    def update_cdf_plots(self, m1f, m1r, m2f, m2r, m3f, m3r):
        """Update CDF plots with current data"""
        # Clear previous plots
        self.fidelity_plot.clear()
        self.rate_plot.clear()

        # Only plot if we have data
        if m1f and m2f and m3f:
            # Fidelity CDF
            m1fx = np.sort(m1f)
            m1fy = np.arange(len(m1fx)) / float(len(m1fx))
            m2fx = np.sort(m2f)
            m2fy = np.arange(len(m2fx)) / float(len(m2fx))
            m3fx = np.sort(m3f)
            m3fy = np.arange(len(m3fx)) / float(len(m3fx))

            # Rate CDF with log10 transformation
            # Handle zero rates by adding small epsilon to avoid log10(0)
            epsilon = 1e-10
            m1rx = np.sort([np.log10(max(r, epsilon)) for r in m1r])
            m1ry = np.arange(len(m1rx)) / float(len(m1rx))
            m2rx = np.sort([np.log10(max(r, epsilon)) for r in m2r])
            m2ry = np.arange(len(m2rx)) / float(len(m2rx))
            m3rx = np.sort([np.log10(max(r, epsilon)) for r in m3r])
            m3ry = np.arange(len(m3rx)) / float(len(m3rx))

            # Plot fidelity
            self.fidelity_plot.plot(m1fx, m1fy, pen=pg.mkPen('blue', width=2), name="R Priority")
            self.fidelity_plot.plot(m2fx, m2fy, pen=pg.mkPen('green', width=2, style=Qt.DashLine), name="Balance")
            self.fidelity_plot.plot(m3fx, m3fy, pen=pg.mkPen('red', width=2, style=Qt.DotLine), name="F Priority")
            self.fidelity_plot.addLegend()

            # Plot rate (log10 scale)
            self.rate_plot.plot(m1rx, m1ry, pen=pg.mkPen('blue', width=2), name="R Priority")
            self.rate_plot.plot(m2rx, m2ry, pen=pg.mkPen('green', width=2, style=Qt.DashLine), name="Balanced")
            self.rate_plot.plot(m3rx, m3ry, pen=pg.mkPen('red', width=2, style=Qt.DotLine), name="F Priority")
            self.rate_plot.addLegend()

    def update_scatter_plot(self, m1f, m1r, m2f, m2r, m3f, m3r):
        """Update scatter plot with current data (using log10 for rate)"""
        # Remove old scatter items
        for mode in [1, 2, 3]:
            if mode in self.scatter_items:
                self.scatter_plot.removeItem(self.scatter_items[mode])

        # Add new scatter plots with log10 transformation
        colors = ['blue', 'green', 'red']
        symbols = ['o', 's', 't']
        mode_names = ["R Priority", "Balanced", "F Priority"]
        modes_data = [(m1f, m1r), (m2f, m2r), (m3f, m3r)]

        # Handle zero rates by adding small epsilon to avoid log10(0)
        epsilon = 1e-10

        for i, (fid_data, rate_data) in enumerate(modes_data):
            if fid_data and rate_data:
                log_rate_data = [np.log10(max(r, epsilon)) for r in rate_data]
                scatter = pg.ScatterPlotItem(
                    x=fid_data, y=log_rate_data,
                    pen=pg.mkPen(colors[i], width=1),
                    brush=pg.mkBrush(colors[i]),
                    symbol=symbols[i],
                    size=5,
                    name=mode_names[i]
                )
                self.scatter_plot.addItem(scatter)
                self.scatter_items[i + 1] = scatter

        # Add legend with white background at bottom left
        legend = self.scatter_plot.addLegend(offset=(10, -10))  # Position at top left
        legend.setBrush(pg.mkBrush(255, 255, 255, 255))  # Solid white background
        legend.setPen(pg.mkPen(0, 0, 0, 255))  # Black border


    def visualize_graph(self, G):
        """Visualize the network graph (placeholder for future implementation)"""
        self.log_message(f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    def simulation_done(self, m1f, m1r, m2f, m2r, m3f, m3r, fname):
        """Handle simulation completion"""
        self.results = (m1f, m1r, m2f, m2r, m3f, m3r, fname)

        # Final plot update
        self.update_cdf_plots(m1f, m1r, m2f, m2r, m3f, m3r)
        self.update_scatter_plot(m1f, m1r, m2f, m2r, m3f, m3r)

        # Update UI - re-enable inputs
        self.set_inputs_enabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)
        self.reset_button.setEnabled(True)

        self.log_message(f"Simulation completed. Results saved to {fname}")

    def save_results(self):
        """Save the results to a file manually"""
        if not self.results:
            return

        m1f, m1r, m2f, m2r, m3f, m3r, fname = self.results

        # Get save path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", fname, "Data Files (*.dat);;All Files (*)"
        )

        if file_path:
            try:
                # Prepare data for saving
                m1fx = np.sort(m1f)
                m1fy = np.arange(len(m1fx)) / float(len(m1fx))
                m2fx = np.sort(m2f)
                m2fy = np.arange(len(m2fx)) / float(len(m2fx))
                m3fx = np.sort(m3f)
                m3fy = np.arange(len(m3fx)) / float(len(m3fx))

                m1rx = np.sort(m1r)
                m1ry = np.arange(len(m1rx)) / float(len(m1rx))
                m2rx = np.sort(m2r)
                m2ry = np.arange(len(m2rx)) / float(len(m2rx))
                m3rx = np.sort(m3r)
                m3ry = np.arange(len(m3rx)) / float(len(m3rx))

                data = np.array([m1rx, m1ry, m2rx, m2ry, m3rx, m3ry,
                                 m1fx, m1fy, m2fx, m2fy, m3fx, m3fy]).T

                np.savetxt(file_path, data, delimiter="\t",
                           header="X1\tY1\tX2\tY2\tX3\tY3\tX4\tY4\tX5\tY5\tX6\tY6", comments="")

                self.log_message(f"Results manually saved to {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    multiprocessing.freeze_support()

    # Create application
    app = QApplication(sys.argv)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec_())
