import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import os

#objective_functions
from objective_functions import objective_functions_configurations

#Algorithms
from robust_algorithms import algorithms


#file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"results_{timestamp}"
# csv_filename = f"algorithm_results_{timestamp}.csv"

#create folder for results if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

headers = ["Objective Function", "MFE", "Best Fitness",]
success_rate_headers = ["Objective Function", "Success Rate"]

cma_es_summary_data = []
rcga_p_summary_data = []

 # saving average fitness history
average_fitness_history = []

best_fitness_history = []



# === User Config Storage ===
user_config = {
    "selected_objectives": [],
    "selected_algorithms": [],
    "dimensions": None,
    "generations": None
}

def success_rate():
    # clear previous output
    output_text.delete(1.0, tk.END)
    
    # Collect selected objective function names
    selected_obj_indices = obj_listbox.curselection()
    selected_obj_names = [obj_listbox.get(i) for i in selected_obj_indices]

    # Store full config objects for selected objective functions
    user_config["selected_objectives"] = [objective_functions_configurations[name] for name in selected_obj_names]

    # Collect algorithms
    selected_algo_indices = algo_listbox.curselection()
    selected_algo_names = [algo_listbox.get(i) for i in selected_algo_indices]

    user_config["selected_algorithms"] = [algorithms[i] for i in selected_algo_names]

    # Get numeric fields
    try:
        user_config["dimensions"] = int(dim_entry.get())
        user_config["generations"] = int(gen_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Dimensions and Generations must be integers.")
        return
    
    success_rates_for_cmaes = []
    success_rates_for_rcgap = []

    # === Debug Printout ===
    print("\n✅ User Configuration:")
    print("Selected Objectives:")
    for obj in user_config["selected_objectives"]:
        print(f"  → {obj['objective_function'].__name__}, bounds={obj['bounds']}, min={obj['minimum']}")
    print("Algorithms:")


    for algo in user_config["selected_algorithms"]:
        print(f"  → {algo['name']}")
        if algo['name'] == "CMA-ES":
            print(f"starting success rate calculations for {algo['name']}")
            for obj in user_config["selected_objectives"]:
                success_rate = 0
                for i in range(100):
                    test = algo['function'](
                        objective_func=obj['objective_function'].fitness_function,
                        OBJECTIVE_FUNCTION=obj['objective_function'].__name__,
                        bounds=obj["bounds"],
                        N=user_config["dimensions"],
                        gen=user_config["generations"],
                        sigma=obj['step_size']
                    )
                    if test[1] < 10**-5:
                        success_rate += 1
                success_rate_percentage = (success_rate / 100) * 100
                success_rates_for_cmaes.append([obj['objective_function'].__name__, success_rate_percentage])
                output_text.insert(tk.END, f"Success Rate for {algo['name']} on {obj['objective_function'].__name__}: {success_rate_percentage:.2f}%\n")

        if algo['name'] == "RCGA-P":
            print(f"starting success rate calculations for {algo['name']}")
            for obj in user_config["selected_objectives"]:
                success_rate = 0
                for i in range(100):
                    test = algo['function'](
                        calculate_fitness=obj['objective_function'].calculate_fitness,
                        OBJECTIVE_FUNCTION=obj['objective_function'].__name__,
                        pop_size=100,
                        num_dimensions=user_config["dimensions"],
                        bounds=obj['bounds'],
                        generations=user_config["generations"],
                        crossover_rate=0.8,
                        mutation_rate=0.1,
                        elitism_count=5
                    )
                    if test[1] < 10**-5:
                        success_rate += 1
                success_rate_percentage = (success_rate / 100) * 100
                success_rates_for_rcgap.append([obj['objective_function'].__name__, success_rate_percentage])
                output_text.insert(tk.END, f"Success Rate for {algo['name']} on {obj['objective_function'].__name__}: {success_rate_percentage:.2f}%\n")

        cma_sr = pd.DataFrame(success_rates_for_cmaes, columns=success_rate_headers)
        rcga_sr = pd.DataFrame(success_rates_for_rcgap, columns=success_rate_headers)

        success_rate_comparison = pd.DataFrame({
            "Objective Function": cma_sr["Objective Function"],
            "CMA-ES Success Rate": cma_sr["Success Rate"],
            "RCGA-P Success Rate": rcga_sr["Success Rate"]
        })

        # Define file path inside the folder
        file_path = os.path.join(folder_name, "success_rate_comparison.xlsx")
        with pd.ExcelWriter(file_path) as writer:
            success_rate_comparison.to_excel(writer, sheet_name="Success Rate Comparison", index=False)

        
        

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
                best_individuals = []
                function_evals_list = []
                fitness_histories = []
                for i in range(5):
                    print(i + 1)
                    start_time = datetime.now()
                    best_ind, best_val, fitness_history, function_evaluations = algo['function'](obj['objective_function'].fitness_function, OBJECTIVE_FUNCTION = obj['objective_function'].__name__, bounds = obj["bounds"],  N=user_config["dimensions"], gen=user_config["generations"], sigma= obj['step_size']) 
                    end_time = datetime.now()

                    # output_text.insert(tk.END, f"Run {i + 1} — {algo['name']} on {obj['objective_function'].__name__}\n")
                    # output_text.insert(tk.END, f"Best Individual: {np.round(best_ind, 6)}\n")
                    # output_text.insert(tk.END, f"Minimum Value: {np.round(best_val, 6)}\n")
                    # output_text.insert(tk.END, f"time taken: {end_time - start_time}\n")
                    # print(f"Total Time for {algo['name']}: {end_time - start_time}")
                    # print(f"Function Evaluations: {function_evaluations}")
                    # output_text.see(tk.END)  # scroll to bottom

                    best_fitness.append(best_val)
                    best_individuals.append(best_ind)
                    function_evals_list.append(function_evaluations)
                    fitness_histories.append(fitness_history)

                    # generations = list(range(1, len(fitness_history) + 1))


                    #ploting graph
                    # fig = Figure(figsize=(4, 3), dpi=100)
                    # ax = fig.add_subplot(111)
                    # ax.plot(generations, fitness_history, marker='o')
                    # ax.set_title(f'Run {i+1}: {algo["name"]} for {obj["objective_function"].__name__}')
                    # ax.set_xlabel('Generation')
                    # ax.set_ylabel('Best Fitness')
                    # ax.grid(True)

                    # fig.tight_layout()

                    #  # Create and place canvas in Tkinter
                    # canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                    # canvas.draw()
                    # canvas.get_tk_widget().pack(pady=10)

                mfe = np.mean(function_evals_list)
                best_run_index = np.argmax(best_fitness)
                best_of_best = best_individuals[best_run_index]
                best_of_best_val = best_fitness[best_run_index]
                best_fitness_history_at_the_end_of_five_runs = fitness_histories[best_run_index]
                cma_es_summary_data.append([obj['objective_function'].__name__, int(mfe), np.round(best_of_best_val, 6)])

                #calculate average fitness per generation
                max_len = max(len(a) for a in fitness_histories)  # longest array length
                padded = np.full((len(fitness_histories), max_len), np.nan)  # fill with NaN

                for i, arr in enumerate(fitness_histories):
                    padded[i, :len(arr)] = arr

                avg_fitness = np.nanmean(padded, axis=0)  # average across runs
                average_fitness_history.append({"algorithm": algo['name'] ,"func": obj['objective_function'].__name__, "history": avg_fitness})

                best_fitness_history.append({"algorithm": algo['name'] ,"func": obj['objective_function'].__name__, "history": best_fitness_history_at_the_end_of_five_runs})




                output_text.insert(tk.END, f"Average Best Fitness over 5 runs: {np.round(np.mean(best_fitness), 6)}\n")
                output_text.insert(tk.END, f"Mean Function Evaluations (MFE): {mfe}\n")
                output_text.insert(tk.END, f"Best-of-the-best solution: {np.round(best_of_best, 6)}\n")
                output_text.insert(tk.END, f"Best-of-the-best fitness: {np.round(best_of_best_val, 6)}\n\n")

            # summary_data.extend(cma_es_summary_data)
        if algo['name'] == "RCGA-P":
            for obj in user_config["selected_objectives"]:
                best_fitness = []
                best_individuals = []
                function_evals_list = []
                fitness_histories = []
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
                    # output_text.insert(tk.END, f"Run {i + 1} — {algo['name']} on {obj['objective_function'].__name__}\n")
                    # output_text.insert(tk.END, f"Best Individual: {np.round(best_ind, 6)}\n")
                    # output_text.insert(tk.END, f"Minimum Value: {np.round(best_val, 6)}\n")
                    # output_text.insert(tk.END, f"time taken: {end_time - start_time}\n")
                    # print(f"Total Time for {algo['name']} : {end_time - start_time}")
                    # print(f"Function Evaluations: {function_evaluations}")
                    # output_text.see(tk.END)  # scroll to bottom

                    best_fitness.append(best_val)
                    best_individuals.append(best_ind)
                    function_evals_list.append(function_evaluations)
                    fitness_histories.append(fitness_history)

                mfe = np.mean(function_evals_list)
                best_run_index = np.argmax(best_fitness)
                best_of_best = best_individuals[best_run_index]
                best_of_best_val = best_fitness[best_run_index]
                best_fitness_history_at_the_end_of_five_runs = fitness_histories[best_run_index]
                rcga_p_summary_data.append([obj['objective_function'].__name__, int(mfe), np.round(best_of_best_val, 6)])

                #calculate average fitness per generation
                max_len = max(len(a) for a in fitness_histories)  # longest array length
                padded = np.full((len(fitness_histories), max_len), np.nan)  # fill with NaN

                for i, arr in enumerate(fitness_histories):
                    padded[i, :len(arr)] = arr

                avg_fitness = np.nanmean(padded, axis=0)  # average across runs
                average_fitness_history.append({"algorithm": algo['name'] ,"func": obj['objective_function'].__name__, "history": avg_fitness})

                best_fitness_history.append({"algorithm": algo['name'] ,"func": obj['objective_function'].__name__, "history": best_fitness_history_at_the_end_of_five_runs})


                
            
                output_text.insert(tk.END, f"Average Best Fitness over 5 runs: {np.round(np.mean(best_fitness), 6)}\n")
                output_text.insert(tk.END, f"Mean Function Evaluations (MFE): {mfe}\n")
                output_text.insert(tk.END, f"Best-of-the-best solution: {np.round(best_of_best, 6)}\n")
                output_text.insert(tk.END, f"Best-of-the-best fitness: {np.round(best_of_best_val, 6)}\n\n")

        #plotting average fitness history

    functions = set(item["func"] for item in average_fitness_history)


    for func in functions:
        # Filter data for this function
        func_data = [item for item in average_fitness_history if item["func"] == func]

        # Separate CMA-ES and RCGA-P
        cma = next((item for item in func_data if item["algorithm"] == "CMA-ES"), None)
        rcga = next((item for item in func_data if item["algorithm"] == "RCGA-P"), None)

        fig = Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)

        if cma and rcga:
            ax.plot(cma["history"], label="CMA-ES", color="blue", linewidth=0.5)
            ax.plot(rcga["history"], label="RCGA-P", color="red", linewidth=0.5)
        else:
            if cma:
                ax.plot(cma["history"], label="CMA-ES", color="blue", linewidth=0.5)
            else:
                ax.plot(rcga["history"], label="RCGA-P", color="red", linewidth=0.5)      
            

        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title(f"Fitness Comparison - {func} for average in five runs")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        graph_path = os.path.join(folder_name, f"avearge_fitness_comparison_{func}.png")
        fig.savefig(graph_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        #Create and place canvas in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    best_fitness_function = set(item["func"] for item in best_fitness_history)
    

    for func in best_fitness_function:
        # Filter data for this function
        func_data = [item for item in average_fitness_history if item["func"] == func]

        # Separate CMA-ES and RCGA-P
        cma = next((item for item in func_data if item["algorithm"] == "CMA-ES"), None)
        rcga = next((item for item in func_data if item["algorithm"] == "RCGA-P"), None)

        fig = Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)

        if cma and rcga:
            ax.plot(cma["history"], label="CMA-ES", color="blue", linewidth=0.5)
            ax.plot(rcga["history"], label="RCGA-P", color="red", linewidth=0.5)
        else:
            if cma:
                ax.plot(cma["history"], label="CMA-ES", color="blue", linewidth=0.5)
            else:
                ax.plot(rcga["history"], label="RCGA-P", color="red", linewidth=0.5)                
            

        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title(f"Fitness Comparison - {func} for best in 5 runs")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        graph_path = os.path.join(folder_name, f"best_fitness_comparison_{func}.png")
        fig.savefig(graph_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        #Create and place canvas in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)



    cma_df = pd.DataFrame(cma_es_summary_data, columns=headers)
    rcga_df = pd.DataFrame(rcga_p_summary_data, columns=headers)

    mfe_comparison = pd.DataFrame({
    "Objective Function": cma_df["Objective Function"],
    "CMA-ES MFE": cma_df["MFE"],
    "RCGA-P MFE": rcga_df["MFE"]
    })

    best_fitness_comparison = pd.DataFrame({
    "Objective Function": cma_df["Objective Function"],
    "CMA-ES Best Fitness": cma_df["Best Fitness"],
    "RCGA-P Best Fitness": rcga_df["Best Fitness"]
    })

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # excel_filename = f"comparison_results_{timestamp}.xlsx"

    # Define file path inside the folder
    file_path = os.path.join(folder_name, "comparison_results.xlsx")

    with pd.ExcelWriter(file_path) as writer:
        mfe_comparison.to_excel(writer, sheet_name="MFE Comparison", index=False)
        best_fitness_comparison.to_excel(writer, sheet_name="Best Fitness Comparison", index=False)

    
    


    # with pd.ExcelWriter("comparison_results.xlsx") as writer:
    #     mfe_comparison.to_excel(writer, sheet_name="MFE Comparison", index=False)
    #     best_fitness_comparison.to_excel(writer, sheet_name="Best Fitness Comparison", index=False)



        
    

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
submit_btn = tk.Button(root, text="compare", command=submit)
submit_btn.grid(row=4, column= 0, columnspan=2, pady=10)

success_rate_btn = tk.Button(root, text="calculate success rate", command=success_rate)
success_rate_btn.grid(row=4, column=2, columnspan=2, pady=10)


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
