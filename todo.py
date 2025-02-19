import json
# [
# {
# "task" : name ,
#  "deadline" : {"date" : date ,
#                "time" : time,
#  "Priority" : Low/Moderate/High,
#  "Status" : Completed / Pending
#  }
# ]

def actions(act):
    if act == 1:
        add()

    if act == 2:
        update()

    if act == 3:
        remove()

    if act == 4:
        mark()

    if act == 5:
        show()


def add():
    with open("To_do_list.json", "r") as to_do_read:
        data = json.load(to_do_read)
    with open("To_do_list.json" , "w") as to_do_write:
        while True:
            task = input("Enter Task Name (Else enter quit): ")
            if task == "quit":
                break
            dline_date = input("Enter Deadline Date: ")
            dline_time = input("Enter Deaadline Time: ")
            priority = input("Enter Priority: ")

            data.append({
                "task" : task,
                "deadline" : {
                    "date" : dline_date,
                    "time" : dline_time,
                },
                "Priority" : priority,
                "Status" : "Pending"
            })
        data = json.dump(data, to_do_write)

def update():
    task = input("Enter task name to update: ").lower()
    with open("To_do_list.json", "r") as to_do_read:
        data = json.load(to_do_read)
    with open("To_do_list.json" , "w") as to_do_write:
        what  = input("Enter which feild you want to update: ")
        for i in range(len(data)):
            if data[i]["task"].lower() == task:
                if what == "deadline":
                    dline_date = input("Enter Deadline Date: ")
                    dline_time = input("Enter Deaadline Time: ")
                    data[i]["deadline"]["date"] = dline_date
                    data[i]["deadline"]["time"] = dline_time
                else:
                    data[i][what] = input(f"Enter new {what}: ")
        data = json.dump(data, to_do_write)


def remove():
    task = input("Enter task name to remove: ").lower()
    with open("To_do_list.json", "r") as to_do_read:
        data = json.load(to_do_read)
    with open("To_do_list.json" , "w") as to_do_write:
        for i in range(len(data)):
            if data[i]["task"].lower() == task:
                del data[i]
                break
        data = json.dump(data, to_do_write)
#
def mark():
    task = input("Enter task name to mark as completed: ").lower()
    with open("To_do_list.json", "r") as to_do_read:
        data = json.load(to_do_read)
    with open("To_do_list.json" , "w") as to_do_write:
        for i in range(len(data)):
            if data[i]["task"].lower() == task:
                data[i]["Status"] = "Completed"
                break

def show():
    with open("To_do_list.json", "r") as to_do_read:
        data = json.load(to_do_read)
    for i in range(len(data)):
        print(f"{i+1}. {data[i]['task']} - {data[i]['deadline']['date']} {data[i]['deadline']['time']} - Priority: {data[i]['Priority']} - Status: {data[i]['Status']}")





print("To-do-list")
while True:
    print("WHAT ACTIONS DO YOU WANT TO DO?")
    print("1.Add items to To-do-list")
    print("2.Update items in To-do-list")
    print("3.Remove items from To-do-list")
    print("4.Mark items as completed")
    print("5.Show To-do-list")
    print("6.Exit")

    act = int(input("Enter your choice: "))

    if act != 6:
        actions(act)
    else:
        break