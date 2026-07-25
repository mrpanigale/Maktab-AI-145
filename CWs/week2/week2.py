"""This is a CLI karvand manager that you can input Your command with a while loop , and each task has its function"""

APP_TITLE = "Karvand JSON Manager -Student Panel"
# imports✔️
# json library will be use for work with json file
import json

# os library will be use for check path to see if json file is exists or not?
import os

# ========================================================================
# karvands.json path ✔️
path = r"E:\Maktab-145\Datasets\karvands.json"

# ========================================================================
# add karvand function✔️


def add_karvand(
    user_id: int,
    full_name: str,
    city: str,
    degree: str,
    field: str,
    skill_name: str,
    skill_level: str,
    skill_score: int,
    path: str,
):
    """This function add persons to the karvands json file
    if the json file already exists
    but if json file not exists this function
      will call creat json function"""

    # full estructure made by generate karvand function
    karvand = generate_karvand_report(
        user_id, full_name, city, degree, field, skill_name, skill_level, skill_score
    )

    # if karvands file exists
    if os.path.exists(path):
        # bootcamp key will not rewrite if file already exists
        new_person = karvand["karvands"][0]
        with open(path, "r", encoding="utf-8") as file:
            karvand_dictionary = json.load(file)
            # call unique id function to check if user id is unique
            unique_id_result = unique_id(user_id, karvand_dictionary)
            if unique_id_result[0]:
                karvand_dictionary["karvands"].append(new_person)
            else:
                print(f"user id is not unique ,user id will be {unique_id_result[1]}")
                new_person["id"] = unique_id_result[1]
                karvand_dictionary["karvands"].append(new_person)

        # over write the karvands.json or Update karvands.json
        with open(path, "w", encoding="utf-8") as file:
            json.dump(karvand_dictionary, file, ensure_ascii=False, indent=4)
            print("karvand added.")
    else:
        # if karvands.json does not exist
        create_json(karvand, path)


# ========================================================================
# unique id function✔️
def unique_id(user_id: int, karvand_dictionary: dict):
    """This function will check if user id is unique ?
    if it's not ! the user id will be biggest id in database + 1"""
    # extract IDs from karvands list in json file .
    id_list = [person["id"] for person in karvand_dictionary["karvands"]]
    if user_id in id_list:
        # if id was not unique user id will be biggest id in database + 1
        return False, max(id_list) + 1
    else:
        return True, None


# ========================================================================
# edit karvand function✔️
def edit_karvand(path: str, user_id: int):
    """this function will edite karvands and find them with their user id"""
    # read the persons from file at first
    karvand_dictionary = json_reader(path)["karvands"]
    # the for loop will looking for user id
    for karvand in karvand_dictionary:
        if karvand["id"] == user_id:
            # karvand will be delete at first
            delet_karvand(path=path, user_id=user_id)
            print("enter new karvand identity: ")
            # with add and input functions the identity will be edited
            (
                user_id,
                full_name,
                city,
                degree,
                field,
                skill_name,
                skill_level,
                skill_score,
            ) = input_identity()
            add_karvand(
                user_id,
                full_name,
                city,
                degree,
                field,
                skill_name,
                skill_level,
                skill_score,
                path=path,
            )
            # return none to break the function to dont print not found message
            return None
    print("karvand not found")


# ========================================================================
# show karvand function ✔️
def show_karvand(path: str):
    """This function will show user all karvands in pretty style like an identity card"""
    data = json_reader(path)
    if not data:
        print("file is empty")
        return
    bootcamp = data.get("bootcamp", {})
    karvands_list = data.get("karvands", [])

    print(f"\n{'#'*40}")
    print(f"🎓 BOOTCAMP: {bootcamp.get('title')} ({bootcamp.get('year')})")
    print(f"{'#'*40}")

    for person in karvands_list:
        print(f"🆔 ID: {person.get('id')}")
        print(f"👤 Name: {person.get('full_name')}")
        print(f"📍 City: {person.get('city')}")

        edu = person.get("education", {})
        print(f"📚 Education: {edu.get('degree')} in {edu.get('field')}")

        print("🛠  Skills:")
        for skill in person.get("skills", []):
            print(
                f"   - {skill.get('name')} | Level: {skill.get('level')} | Score: {skill.get('score')}"
            )

        print(f"{'-'*40}")


# ========================================================================
# search karvand function✔️
def search_karvand(path: str, user_id: int):
    """This function will find karvand by ID"""
    karvands_dictionary = json_reader(path)["karvands"]
    for karvand in karvands_dictionary:
        if karvand["id"] == user_id:
            print(f"🆔 ID: {karvand.get('id')}")
            print(f"👤 Name: {karvand.get('full_name')}")
            print(f"📍 City: {karvand.get('city')}")

            edu = karvand.get("education", {})
            print(f"📚 Education: {edu.get('degree')} in {edu.get('field')}")

            print("🛠  Skills:")
            for skill in karvand.get("skills", []):
                print(
                    f"   - {skill.get('name')} | Level: {skill.get('level')} | Score: {skill.get('score')}"
                )

            print(f"{'-'*40}")
            return None
    print(f"karvands with  <id : {user_id}> not found")


# ========================================================================
# delet karvand function✔️
def delet_karvand(path: str, user_id: int):
    """This function with user id will find the karvand and delete karvand
    , after this function will rewrite the file with out deleted karvand"""
    karvand_dictionary = json_reader(path)
    karvands = karvand_dictionary["karvands"]
    for i, karvand in enumerate(karvands):
        if karvand["id"] == user_id:
            del karvands[i]
            print("Karvand deleted")
            create_json(karvand_dictionary, path)
            return None
    print("karvand not found")


# ========================================================================
# report all karavands function
def report_karvands(path: str):
    """This function will tell us how many karvands exist in Bootcamp."""
    karvands = json_reader(path)["karvands"]
    print(
        f"\n🟰🟰🟰🟰🟰🟰\n{len(karvands)} karvands exist in This Bootcamp!\n🟰🟰🟰🟰🟰🟰"
    )


# ========================================================================
# read json files function✔️


def json_reader(path: str):
    """This function will read karvands.json
    but if json file does not exist ,
    this function will return an empty structure"""

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            karvands_dictionary = json.load(file)
            return karvands_dictionary
    else:
        print("karvands.json did not creat yet!")
        return {"bootcamp": {}, "karvands": []}


# ========================================================================
# create json file function✔️
def create_json(karvand: dict, path: str):
    """This function will create karvands.json
    if while adding  user karvands.json does not exist"""

    # create and write on karvands.json for first time
    with open(path, "w", encoding="utf-8") as file:
        json.dump(karvand, file, ensure_ascii=False, indent=4)
        print("file craeted and karvand added to the file!")


# ========================================================================
# input function ✔️
def input_identity():
    """This function will get user identity with Error handeling parts
    that make sure user will enter only numeric values for id and skill score"""
    # inputs
    while True:
        try:
            user_id = int(input("user id: "))
            break
        except ValueError:
            print("id should be numeric!")
    full_name = input("\n🟰🟰🟰🟰🟰🟰\nfull name: ")
    city = input("\n🟰🟰🟰🟰🟰🟰\ncity: ")
    degree = input("\n🟰🟰🟰🟰🟰🟰\ndegree (like Bachelor): ")
    field = input("\n🟰🟰🟰🟰🟰🟰\nfield (like Computer Engineering)").title()
    skill_name = input("\n🟰🟰🟰🟰🟰🟰\nskill name: (like Python): ").title()
    skill_level = input("\n🟰🟰🟰🟰🟰🟰\nskill level (like beginner): ")
    while True:
        try:
            skill_score = int(input("\nskill score (like 70): "))
            break
        except ValueError:
            print("only numeric value for score!")

    return user_id, full_name, city, degree, field, skill_name, skill_level, skill_score


# ========================================================================
# generate karavands data✔️
def generate_karvand_report(
    user_id,
    full_name,
    city,
    degree,
    field,
    skill_name,
    skill_level,
    skill_score,
    bootcamp_title="Karvand Python",
    bootcamp_year="2026",
):
    """This function generate karvands.json structure"""

    data = {
        "bootcamp": {"title": bootcamp_title, "year": bootcamp_year},
        "karvands": [
            {
                "id": user_id,
                "full_name": full_name,
                "city": city,
                "education": {"degree": degree, "field": field},
                "skills": [
                    {"name": skill_name, "level": skill_level, "score": skill_score}
                ],
            }
        ],
    }
    return data


# ========================================================================

# loop user control panel✔️

while True:
    # flag
    flag = input("1.Add 2.edit 3.show 4.delet 5.report 6.search 7.exit: ")
    # options
    # add✔️
    if flag == "1":
        (
            user_id,
            full_name,
            city,
            degree,
            field,
            skill_name,
            skill_level,
            skill_score,
        ) = input_identity()
        add_karvand(
            user_id,
            full_name,
            city,
            degree,
            field,
            skill_name,
            skill_level,
            skill_score,
            path=path,
        )

    # edit✔️
    elif flag == "2":
        show_karvand(path=path)
        print("from the list find the user id that you are looking for: ")
        try:
            user_id = int(input("Enter user id"))
        except ValueError:
            print("Only numeric value for id !!!!")
            user_id = int(input("Enter user id"))

        edit_karvand(path=path, user_id=user_id)
    # show✔️
    elif flag == "3":
        show_karvand(path=path)
    # delet✔️
    elif flag == "4":
        try:
            user_id = int(input("Enter user id: "))
            delet_karvand(path, user_id)
        except ValueError:
            print("Only numeric value for id !!!!")
            user_id = int(input("Enter user id: "))
            delet_karvand(path, user_id)
    # report✔️
    elif flag == "5":
        report_karvands(path=path)
    # search id
    elif flag == "6":
        try:
            user_id = int(input("Enter user id: "))
            search_karvand(path, user_id)
        except ValueError:
            print("Only numeric value for id !!!!")
            user_id = int(input("Enter user id: "))
            search_karvand(path, user_id)
    # exit✔️
    elif flag == "7":
        break
    # unknown input✔️
    else:
        print("Unknown input!")
