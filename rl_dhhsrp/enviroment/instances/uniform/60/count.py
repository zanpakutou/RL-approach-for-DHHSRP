cnt = 0
for i in range(500):
    file = open(str(i) + '.in', 'r')
    cnt_day = 0
    cnt_request = 0
    for line in file:
        if "-> Request" in line:
            cnt_request = cnt_request + 1
        if "--> Day" in line:
            cnt_day = cnt_day + 1
    cnt = cnt + cnt_request/cnt_day
cnt = cnt / 500
print(cnt)