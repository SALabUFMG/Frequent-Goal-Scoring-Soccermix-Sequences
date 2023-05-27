def get_pitch_zone(row):
    if row['x'] < 20:
        if row['y'] <= 14:
            return 3
        elif row['y'] <= 54:
            return 2
        else:
            return 1
    elif row['x'] < 36.25:
        if row['y'] <= 14:
            return 8
        elif row['y'] <= 25:
            return 7
        elif row['y'] <= 43:
            return 6
        elif row['y'] <= 54:
            return 5
        else:
            return 4
    elif row['x'] < 52.5:
        if row['y'] <= 14:
            return 13
        elif row['y'] <= 25:
            return 12
        elif row['y'] <= 43:
            return 11
        elif row['y'] <= 54:
            return 10
        else:
            return 9
    elif row['x'] < 68.75:
        if row['y'] <= 14:
            return 18
        elif row['y'] <= 25:
            return 17
        elif row['y'] <= 43:
            return 16
        elif row['y'] <= 54:
            return 15
        else:
            return 14
    elif row['x'] < 85:
        if row['y'] <= 14:
            return 23
        elif row['y'] <= 25:
            return 22
        elif row['y'] <= 43:
            return 21
        elif row['y'] <= 54:
            return 20
        else:
            return 19
    else:
        if row['y'] <= 14:
            return 28
        elif row['y'] <= 25:
            return 27
        elif row['y'] <= 43:
            return 26
        elif row['y'] <= 54:
            return 25
        else:
            return 24