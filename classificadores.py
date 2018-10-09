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

def main():
    dados, labels = PegaDados()
    kmeans = k_means(dados)
    print(kmeans.labels_)
    print()
    print(labels)
    print()
    print("Acuracia: ",np.sum(labels == kmeans.labels_)/labels.shape[0])

main()

