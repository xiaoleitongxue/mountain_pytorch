class Coordinate:
    x = 0
    y = 0

    def __init__(self, x_, y_):
        x = x_
        y = y_

def local_coordinates2global_coordinates(offset_x, offset_y, coordinate):
    coordinate.x = coordinate.x + offset_x
    coordinate.y = coordinate.y + offset_y
    return coordinate


def batch_transformer(coordinates):
    list = []
    for coordinate in coordinates:
        transed_coor = local_coordinates2global_coordinates(offset_x, offset_y, coordinate)
        list.append(transed_coor)
    return list

