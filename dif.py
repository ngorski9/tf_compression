if __name__ == "__main__":
    lines1 = []
    lines2 = []

    inf1 = open("../old.txt", "r")
    inf2 = open("../new.txt", "r")

    for line in inf1:
        lines1.append(line)

    for line in inf2:
        lines2.append(line)
    
    num = 0

    for i in range(len(lines1)):
        if lines1[i] != lines2[i]:
            num += 1
            print(lines1[i])
            print(lines2[i])
            print("-------")

    print(num)