from pathlib import Path

data_path = Path("data/aclImdb/train")
positive_path = data_path/"pos"
negative_path = data_path / "neg"

print("+ reviews:", len(list(positive_path.glob("*.txt")))) #number of pos reviews
print("- reviews:", len(list(negative_path.glob("*.txt")))) #number of neg reviews

positive_files = list(positive_path.glob("*.txt")) #
negative_files = list(negative_path.glob("*.txt")) 
with open(positive_files[0], "r", encoding="utf-8") as file: positive_review = file.read()
with open(negative_files[0], "r", encoding="utf-8") as file: negative_review = file.read()
print("\n random + review:")
print(positive_review[:500]) #show a pos review
print("\n random - review:")
print(negative_review[:500]) #show a neg review

reviews = []
labels = []
for file_path in positive_files:
    with open(file_path, "r", encoding="utf-8") as file:
        review_text = file.read()
    reviews.append(review_text)
    labels.append(1) #pos loop & store
for file_path in negative_files:
    with open(file_path, "r", encoding="utf-8") as file:
        review_text = file.read()
    reviews.append(review_text)
    labels.append(0)  #neg loop and store
    
print(len(reviews)) #should print 25000
print(len(labels)) #should print 25000
print("first label: ", labels[0])
print("first review: ", reviews[0][:100])
print("last label: ", labels[-1])
print("last review: ", reviews[-1][:100]) #san check