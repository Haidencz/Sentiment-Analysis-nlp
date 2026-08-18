from pathlib import Path

data_path = Path("data/aclImdb/train")
positive_path = data_path/"pos"
negative_path = data_path / "neg"

print("+ reviews:", len(list(positive_path.glob("*.txt")))) 
print("- reviews:", len(list(negative_path.glob("*.txt"))))

positive_files = list(positive_path.glob("*.txt"))
negative_files = list(negative_path.glob("*.txt"))
with open(positive_files[0], "r", encoding="utf-8") as file: positive_review = file.read()
with open(negative_files[0], "r", encoding="utf-8") as file: negative_review = file.read()
print("\n random + review:")
print(positive_review[:500])
print("\n random - review:")
print(negative_review[:500])