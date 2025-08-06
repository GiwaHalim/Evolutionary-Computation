import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#objective_functions
from objective_functions import objective_functions_configurations

#Algorithms
from robust_algorithms import algorithms


# === User Config Storage ===
user_config = {
    "selected_objectives": [],
    "selected_algorithms": [],
    "dimensions": 0,
    "generations": 0
}

def submit():
    #clear previous output
    output_text.delete(1.0, tk.END)
    # Collect selected objective function names
    selected_obj_indices = obj_listbox.curselection()
    selected_obj_names = [obj_listbox.get(i) for i in selected_obj_indices]

    # Store full config objects for selected objective functions
    user_config["selected_objectives"] = [objective_functions_configurations[name] for name in selected_obj_names]

    # Collect algorithms
    selected_algo_indices = algo_listbox.curselection()
    selected_algo_names = [algo_listbox.get(i) for i in selected_algo_indices]

    # print([algorithms(i) for i in selected_algo_names])
    user_config["selected_algorithms"] = [algorithms[i] for i in selected_algo_names]

    # Get numeric fields
    try:
        user_config["dimensions"] = int(dim_entry.get())
        user_config["generations"] = int(gen_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Dimensions and Generations must be integers.")
        return

    # === Debug Printout ===
    print("\n✅ User Configuration:")
    print("Selected Objectives:")
    for obj in user_config["selected_objectives"]:
        print(f"  → {obj['objective_function'].__name__}, bounds={obj['bounds']}, min={obj['minimum']}")
    print("Algorithms:")

    # === Run Selected Algorithms ===
    for algo in user_config["selected_algorithms"]:
        print(f"  → {algo['name']}")
        if algo['name'] == "CMA-ES":
            for obj in user_config["selected_objectives"]:
                best_fitness = []
                for i in range(5):
                    print(i + 1)
                    start_time = datetime.now()
                    best_ind, best_val, fitness_history, function_evaluations = algo['function'](obj['objective_function'].fitness_function, OBJECTIVE_FUNCTION = obj['objective_function'].__name__, bounds = obj["bounds"],  N=user_config["dimensions"], gen=user_config["generations"], sigma= obj['step_size']) 
                    end_time = datetime.now()
                    output_text.insert(tk.END, f"Run {i + 1} — {algo['name']} on {obj['objective_function'].__name__}\n")
                    output_text.insert(tk.END, f"Best Individual: {np.round(best_ind, 6)}\n")
                    output_text.insert(tk.END, f"Minimum Value: {np.round(best_val, 6)}\n")
                    output_text.insert(tk.END, f"time taken: {end_time - start_time}\n")
                    print(f"Total Time for {algo['name']}: {end_time - start_time}")
                    print(f"Function Evaluations: {function_evaluations}")
                    # output_text.insert(tk.END, f"Fitness History: {np.round(fitness_history, 6)}\n\n")
                    output_text.see(tk.END)  # scroll to bottom
                    best_fitness.append(best_val)
                    generations = list(range(1, len(fitness_history) + 1))

                    #ploting graph
                    fig = Figure(figsize=(4, 3), dpi=100)
                    ax = fig.add_subplot(111)
                    ax.plot(generations, fitness_history, marker='o')
                    ax.set_title(f'Run {i+1}: {algo["name"]} for {obj["objective_function"].__name__}')
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Best Fitness')
                    ax.grid(True)

                    fig.tight_layout()

                     # Create and place canvas in Tkinter
                    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(pady=10)
            

                output_text.insert(tk.END, f"Average Best Fitness over 5 runs: {np.round(np.mean(best_fitness), 6)}\n\n")

        if algo['name'] == "RCGA-P":
            for obj in user_config["selected_objectives"]:
                best_fitness = []
                for i in range(5):
                    print(i + 1)
                    start_time = datetime.now()
                    best_ind, best_val, fitness_history, function_evaluations = algo['function'](
                        calculate_fitness=obj['objective_function'].calculate_fitness,
                        OBJECTIVE_FUNCTION = obj['objective_function'].__name__,
                        pop_size=100,
                        num_dimensions=user_config["dimensions"],
                        bounds=obj['bounds'],
                        generations=user_config["generations"],
                        crossover_rate=0.8,
                        mutation_rate=0.1,
                        elitism_count=5
                    )
                    end_time = datetime.now()
                    output_text.insert(tk.END, f"Run {i + 1} — {algo['name']} on {obj['objective_function'].__name__}\n")
                    output_text.insert(tk.END, f"Best Individual: {np.round(best_ind, 6)}\n")
                    output_text.insert(tk.END, f"Minimum Value: {np.round(best_val, 6)}\n")
                    output_text.insert(tk.END, f"time taken: {end_time - start_time}\n")
                    best_fitness.append(best_val)
                    print(f"Total Time for {algo['name']} : {end_time - start_time}")
                    print(f"Function Evaluations: {function_evaluations}")
                    # output_text.insert(tk.END, f"Fitness History: {np.round(fitness_history, 6)}\n\n")
                    output_text.see(tk.END)  # scroll to bottom
                    generations = list(range(1, len(fitness_history) + 1))

                    #ploting graph
                    fig = Figure(figsize=(4, 3), dpi=100)
                    ax = fig.add_subplot(111)
                    ax.plot(generations, fitness_history, marker='o')
                    ax.set_title(f'Run {i+1}: {algo["name"]} for {obj["objective_function"].__name__}')
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Best Fitness')
                    ax.grid(True)

                    fig.tight_layout()

                    # Create and place canvas in Tkinter
                    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(pady=10)

                output_text.insert(tk.END, f"Average Best Fitness over 5 runs: {np.round(np.mean(best_fitness), 6)}\n\n")


# === Create Tkinter UI ===
root = tk.Tk()
root.title("Optimization Setup")
root.attributes('-fullscreen', True)

# make graph section resizable
plot_container = ttk.Frame(root)
plot_container.grid(row=6, column=0, padx=10, pady=10, sticky="nsew")
root.grid_rowconfigure(6, weight=1)
root.grid_columnconfigure(0, weight=1)


# === Objective Function Listbox ===
tk.Label(root, text="Select Objective Functions (multi-select):").grid(row=0, column=0, sticky="w")
obj_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=5, exportselection=False)
for name in objective_functions_configurations.keys():
    obj_listbox.insert(tk.END, name)
obj_listbox.grid(row=1, column=0, padx=5, pady=5)

# === Algorithm Listbox ===
tk.Label(root, text="Select Algorithms (multi-select):").grid(row=0, column=1, sticky="w")
algo_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=5, exportselection=False)
for algo in algorithms.keys():
    algo_listbox.insert(tk.END, algo)
algo_listbox.grid(row=1, column=1, padx=5, pady=5)

# === Dimensions Input ===
tk.Label(root, text="Number of Dimensions:").grid(row=2, column=0, sticky="w")
dim_entry = tk.Entry(root)
dim_entry.grid(row=3, column=0, padx=5, pady=5)

# === Generations Input ===
tk.Label(root, text="Number of Generations:").grid(row=2, column=1, sticky="w")
gen_entry = tk.Entry(root)
gen_entry.grid(row=3, column=1, padx=5, pady=5)

# === Submit Button ===
submit_btn = tk.Button(root, text="Submit", command=submit)
submit_btn.grid(row=4, columnspan=2, pady=10)


# printimg results 
output_text = tk.Text(root, height=20, width=150)
output_text.grid(row = 5, pady=10)

# == graphs ===
plot_canvas = tk.Canvas(plot_container, height=300, width=1000)  # adjust height as needed
plot_canvas.grid(row=6, column=0, sticky="nsew")

scrollbar = ttk.Scrollbar(plot_container, orient="vertical", command=plot_canvas.yview)
scrollbar.grid(row=6, column=1, sticky="ns")

plot_canvas.configure(yscrollcommand=scrollbar.set)

# === Scrollable Frame inside canvas ===
plot_frame = ttk.Frame(plot_canvas)
plot_frame.bind(
    "<Configure>",
    lambda e: plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))
)

# Add the frame to the canvas
plot_canvas.create_window((0, 0), window=plot_frame, anchor="nw")

root.mainloop()
