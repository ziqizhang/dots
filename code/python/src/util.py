import pandas as pd
import csv

def merge_ENCASEH2020_dataset(folder,outfile):
    df_features = pd.read_csv(folder+"/hatespeech_features.csv", sep=',', encoding="utf-8")
    df_labels=pd.read_csv(folder+"/hatespeech_labels.csv", sep=',', encoding="utf-8").as_matrix()

    label_lookup = {}
    for row in df_labels:
        label_lookup[row[0]]=row[1]

    with open(outfile, 'w',newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header=list(df_features.columns.values)
        header.append("class")

        csvwriter.writerow(header)
        df_features=df_features.as_matrix()
        for row in df_features:
            id=row[0]
            try:
                label=label_lookup[int(id)]
                new_row=list(row)
                new_row.append(label)
                csvwriter.writerow(row)
            except ValueError:
                continue


if __name__=="__main__":
    merge_ENCASEH2020_dataset("/home/zz/Work/dots/data/ENCASEH2020/raw",
                              "/home/zz/Work/dots/data/ENCASEH2020/labeled_data_all.csv")