from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn_extensions.fuzzy_kmeans import KMedians, FuzzyKMeans
from sklearn import metrics
from scipy.spatial import distance
from sklearn.cluster import KMeans
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

def separa_conjuntos(kmeans, fuzzy, dados):
    k1, k0, f1, f0 = [], [], [], []
    k = kmeans.labels_
    f = fuzzy.labels_
    for i in range(0, dados.shape[0]):
        if(k[i] == 0):
            k0.append(dados[i])
        else:
            k1.append(dados[i])
        if(f[i] == 0):
            f0.append(dados[i])
        else:
            f1.append(dados[i])

    return k0, k1, f0, f1

def fisher(kmeans, fuzzy, k0, k1, f0, f1):
    distancias_k0 = distance.cdist(k0, kmeans.cluster_centers_[0].reshape(1, -1), 'euclidean')
    distancias_k1 = distance.cdist(k1, kmeans.cluster_centers_[1].reshape(1, -1), 'euclidean')
    distancias_f0 = distance.cdist(f0, fuzzy.cluster_centers_[0].reshape(1, -1), 'euclidean')
    distancias_f1 = distance.cdist(f1, fuzzy.cluster_centers_[1].reshape(1, -1), 'euclidean')
    
    fisherK = (np.power((np.mean(distancias_k0) - np.mean(distancias_k1)),2))/(np.power(np.std(distancias_k0),2) + np.power(np.std(distancias_k1),2))
    fisherF = (np.power((np.mean(distancias_f0) - np.mean(distancias_f1)),2))/(np.power(np.std(distancias_f0),2) + np.power(np.std(distancias_f1),2))

    return fisherK, fisherF

def main():
    dados, labels = PegaDados()
    kmeans = k_means(dados)
    fuzzy = fuzzy_means(dados)
    acuraciak, acuraciaF = comp_labels(kmeans, fuzzy, labels)
    metrica_sil_K, metrica_sil_F = comp_silhuetta(kmeans, fuzzy, dados)
    k0, k1, f0, f1 = separa_conjuntos(kmeans, fuzzy, dados)
    fisherK, fisherF = fisher(kmeans, fuzzy, k0, k1, f0, f1)
    
    print("\nLabels gerados pelo K-means\n")
    print(kmeans.labels_)
    print("\nLabels gerados pelo Fuzzy-means\n")
    print(fuzzy.labels_)
    print("\nLabels originais do dataset\n")
    print(labels)
    print()
    print("Critério de fisher com k-means: ",fisherK)
    print("Metrica silhuetta k-means: ",metrica_sil_K)
    print("Acuracia K-means: ",acuraciak)
    print()
    print("Critério de fisher com Fuzzy-means: ",fisherF)
    print("Metrica silhuetta Fuzzy-means: ",metrica_sil_F)
    print("Acuracia Fuzzy-means: ", acuraciaF)

main()

