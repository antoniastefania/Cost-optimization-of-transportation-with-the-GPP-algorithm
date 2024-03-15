import numpy as np

def gauss_pivotare_partiala(A, b):
    n = len(b)

    # Construim matricea extinsă
    matrice_extinsa = np.column_stack((A.astype(np.float64), b.astype(np.float64)))

    print("Matricea extinsa initiala:\n", matrice_extinsa)

    # Eliminare gaussiana cu pivotare partiala
    for i in range(n-1):
        # Alegem pivotul
        pivot_index = np.argmax(np.abs(matrice_extinsa[i:, i])) + i

        # Schimbăm rândurile pentru a avea pivotul pe diagonala principală
        matrice_extinsa[[i, pivot_index]] = matrice_extinsa[[pivot_index, i]]

        # Eliminare gaussiana
        pivot = matrice_extinsa[i, i]
        if pivot == 0:
            # Poate apărea când toate elementele sub pivot sunt deja zero
            continue

        matrice_extinsa[i] /= pivot

        for j in range(i+1, n):
            ratio = matrice_extinsa[j, i]
            matrice_extinsa[j] -= ratio * matrice_extinsa[i]

    print("\nMatricea extinsa dupa eliminarea gaussiana:\n", matrice_extinsa)

    # Substituție înapoi
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = matrice_extinsa[i, -1] - np.dot(matrice_extinsa[i, i+1:n], x[i+1:])

    return x

# Generare date aleatoare
np.random.seed(42)  # Setăm o sămânță pentru reproducibilitate

# Definire liste pentru produse, producători și destinații
nume_produse = ["Pâine", "Ouă", "Lapte de vaca", "Mere", "Castraveti", "Carne tocata amestec vita-porc,", "Peste", "Paste", "Orez", "Ulei", "Ciocolata", "Suc", "Apa", "Prajituri", "Chipsuri", "Pufuleti", "Branza", "Cascaval", "Mozzarella", "Parmezan", "Lapte de soia", "Lapte de oaie", "Lapte de capra", "Rosii", "Portocale", "Pomelo", "Ardei", "Patrunjel", "Marar", "Cartofi albi", "Cartofi dulci", "Piept de pui", "Pulpe de pui", "Carne de porc", "Carne de vitel", "Pastrama", "Carnati grosi", "Carnati subtiri", "Varza alba", "Varza rosie", "Leustean", "Cafea", "Cereale", "Ceapa", "Usturoi", "Condimente", "Chifle", "Fistic", "Nuca", "Arahide", "Iaurt", "Zahar", "Faina", "Carne tocata de pui", "Carne de curcan", "Lamai"]

nume_producatori = ["AgroDeligts", "BioHarvest", "FreshFoods Co.", "GourmetGoodies", "Organica Farms", "Velpitar", "Bunica", "Covalact", "Napolact", "Delaco", "Nestle", "OuaFericite Ferma", "PuiRomanesc", "CarnatiAutentici", "OaiaFericita Ferma", "Dulciuri Romanesti", "Bucuria", "Chio", "Lays", "Olympus", "Plantatie legume si fructe Spania", "Made in Turkey", "From Italy", "DeliciiDeCacao", "UleiDeSolei", "Barilla", "RosiiRomanesti Plantatie", "Fragedo", "Cocorico", "Nescafe Dolce Gusto", "Tchibo", "Alpro", "Fortuna Randez-vous", "AromeRomanesti Fabrica", "DeliKat", "Grania", "VarzaNoastra Plantatie", "SucNatural SRL", "IzvorRomanesc SA", "Aqua Carpatica", "Borsec"]

nume_destinatii = ["Mega Image", "Carrefour", "Auchan", "Kaufland", "Cora", "La Cocos", "La doi pasi", "Profi", "Lidl"]

# Amestecarea listelor
np.random.shuffle(nume_produse)
np.random.shuffle(nume_producatori)
np.random.shuffle(nume_destinatii)

# Ajustează listele pentru a fi utilizate ca elemente distincte în matrice
matrice_produse = np.random.choice(nume_produse, size=(2000, 2000))
matrice_producatori = np.random.choice(nume_producatori, size=(2000, 2000))
matrice_destinatii = np.random.choice(nume_destinatii, size=(2000, 2000))

# Număr de producători, destinatii și produse
numar_producatori = 2000
numar_destinatii = 2000
numar_produse = 2000


# Generare matrice de costuri între producători și destinatii
costuri_transport = np.random.randint(1, 10, size=(numar_producatori, numar_destinatii))

# Generare vector de oferte (cantitatea disponibilă la fiecare producător)
oferte = np.random.randint(150, 200, size=numar_producatori)

# Generare vector de cereri (cantitatea necesară la fiecare destinație)
cereri = np.random.randint(60, 140, size=numar_destinatii)
# Rezolvarea sistemului liniar folosind Gauss cu pivotare parțială
sol = gauss_pivotare_partiala(costuri_transport, cereri)

# Afișează rezultatele în fișiere separate
np.savetxt('costuri_de_transport.txt', costuri_transport, fmt='%d')

# Salvare cantitati_optime_numpy.txt (fără transformare în zero)
np.savetxt('cantitati_optime_numpy.txt', sol, fmt='%.5f')

# Creare matrice cu informații despre cantitățile indisponibile
matrice_indisponibile = []

for i in range(numar_destinatii):
    for j in range(numar_producatori):
        index = i * numar_producatori + j  # Indexare corectă în vectorul 1-dimensional
        if sol[j] <= 0:  # Produsele cu cantitate negativă sau zero
            produs = matrice_produse[j, i]
            oferta = oferte[j]
            cerere = cereri[i]
            producator = matrice_producatori[j, i]
            destinatie = matrice_destinatii[j, i]
            matrice_indisponibile.append([produs, oferta, cerere, producator, destinatie])

# Salvare în fișierul cantitati_indisponibile.txt
with open('cantitati_indisponibile.txt', 'w', encoding='utf-8') as file:
    file.write("Produs\tOferta\tCerere\tProducator\tDestinatie\n")
    for row in matrice_indisponibile:
        file.write('\t'.join(map(str, row)) + '\n')


