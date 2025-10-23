import pandas as pd
import math
import os
import shutil

main_path_base = "../pooling_paper_results/google_drive_results/cl_more_epochs"
main_path_large = "../pooling_paper_results/google_drive_results/large"

cl_main_path = "../pooling_paper_results/tables_final_results/CL_more_epochs"
destino_cl_acc = "../pooling_paper_results/FILES/cl_all_acc_files"
destino_cl_devacc = "../pooling_paper_results/FILES/cl_all_devacc_files"

os.makedirs(destino_cl_acc, exist_ok=True)
os.makedirs(destino_cl_devacc, exist_ok=True)

# Percorrer todas as subpastas
for root, dirs, files in os.walk(cl_main_path):
    # Verificar se está na pasta "acc"
    if os.path.basename(root) == "cl_acc":
        for file in files:
            caminho_origem = os.path.join(root, file)
            shutil.copy(caminho_origem, destino_cl_acc)
            #print(f"Copiado: {caminho_origem} para {destino_cl_acc}")

    # Verificar se está na pasta "dev"
    elif os.path.basename(root) == "cl_devacc":
        for file in files:
            caminho_origem = os.path.join(root, file)
            shutil.copy(caminho_origem, destino_cl_devacc)
            #print(f"Copiado: {caminho_origem} para {destino_cl_devacc}")

#print("Arquivos copiados com sucesso!")


#####FILES BASE
files_base_acc = [destino_cl_acc + "/" + p for p in os.listdir(destino_cl_acc)]
files_base_devacc = [destino_cl_devacc + "/" + p for p in os.listdir(destino_cl_devacc)]

######FILES LARGE
files_large_acc = []
files_large_devacc = []

#JOINS TABLES BASE
if len(files_base_acc) > 0 and len(files_base_devacc) > 0:

    #ACC
    dataframes_base_acc = [pd.read_csv(file) for file in files_base_acc]
    combined_df_base_acc = pd.concat(dataframes_base_acc, ignore_index=True)
    combined_df_base_acc.to_csv(main_path_base + "/resultados_cl_base_acc.csv", index=False)

    combined_df_base_acc = combined_df_base_acc.map(
        lambda x: f"{x:.2f}".replace(".", ",") if isinstance(x, (float, int)) else x
    )
    combined_df_base_acc.to_csv(main_path_base + "/resultados_cl_base_acc_google_drive.csv", index=False, sep=";")

    #DEVACC
    dataframes_base_devacc = [pd.read_csv(file) for file in files_base_devacc]
    combined_df_base_devacc = pd.concat(dataframes_base_devacc, ignore_index=True)
    combined_df_base_devacc.to_csv(main_path_base + "/resultados_cl_base_devacc.csv", index=False)

    combined_df_base_devacc = combined_df_base_devacc.map(
        lambda x: f"{x:.2f}".replace(".", ",") if isinstance(x, (float, int)) else x
    )
    combined_df_base_devacc.to_csv(main_path_base + "/resultados_cl_base_devacc_google_drive.csv", index=False, sep=";")