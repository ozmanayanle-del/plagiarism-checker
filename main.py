from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("document plagiarism checker")
print("===========================")
document1 = input("geli qoraalka kowad: ")
document2 = input("geli qoraalka kowad: ")

documents = [ document1, document2]
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(documents)
similarty = cosine_similarity(matrix[0:1], matrix[0:1])[0][0]
percentage = similarty * 100
print("\nNATIIJADA")
print("===========")
print(f"isu ekaansho qoraallada: {percentage :.2f}%")

if percentage > 80:
    print("isu ekaansho sare")
elif percentage > 40:
    print("isu ekaansho dhexdhexaad")
else:
    print("isu ekansho hoose")

print("\n mahadsanid isticmaalka barnaamijkan")
