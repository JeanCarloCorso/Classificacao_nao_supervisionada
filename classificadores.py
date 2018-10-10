from sklearn.cluster import KMeans
from sklearn_extensions.fuzzy_kmeans import KMedians, FuzzyKMeans
from sklearn import metrics
import numpy as np

def PegaDados():
    dados = np.loadtxt("cancer.data", delimiter=",") # pega o dataset
    label_bruto = open("cancer-label.data", 'r')

    label = np.zeros(569).reshape((569))
    c = 0
    for l in label_bruto:
        #print(l)
        if(l == "M\n"):
            #print("entrou M")
            label[c] = 1
        elif(l == "B\n"):
            #print("entrou B")
            label[c] = 0
        c = c + 1
    label = label.astype(int)
    
    return dados, label

def k_means(dados):
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit(dados)

    return kmeans

def fuzzy_means(dados):
    fuzzy_kmeans = FuzzyKMeans(k=2, m=1000)
    fuzzy_kmeans.fit(dados)
    
    return fuzzy_kmeans

def comp_labels(kmeans, fuzzy, labels):
    acuraciak = np.sum(labels == kmeans.labels_)/labels.shape[0]
    acuraciaF = np.sum(labels == fuzzy.labels_)/labels.shape[0]

    return acuraciak, acuraciaF

def comp_silhuetta(kmeans, fuzzy, dados, metric='euclidean'):
    metrica_sil_K = metrics.silhouette_score(dados, kmeans.labels_, metric)
    metrica_sil_F = metrics.silhouette_score(dados, fuzzy.labels_, metric)

    return metrica_sil_K, metrica_sil_F

def main():
    dados, labels = PegaDados()
    kmeans = k_means(dados)
    fuzzy = fuzzy_means(dados)
    acuraciak, acuraciaF = comp_labels(kmeans, fuzzy, labels)
    metrica_sil_K, metrica_sil_F = comp_silhuetta(kmeans, fuzzy, dados)
    print("\nLabels gerados pelo K-means\n")
    print(kmeans.labels_)
    print("\nLabels gerados pelo Fuzzy-means\n")
    print(fuzzy.labels_)
    print("\nLabels originais do dataset\n")
    print(labels)
    print()
    print("Metrica silhuetta k-means: ",metrica_sil_K)
    print("Acuracia K-means: ",acuraciak)
    print()
    print("Metrica silhuetta Fuzzy-means: ",metrica_sil_F)
    print("Acuracia Fuzzy-means: ", acuraciaF)

main()

