# making vertices of the trianlge to be in counterclockwise direction
def order_vertices_ccw(v1, v2, v3):
    # Calculate the cross product of the vectors (v2 - v1) and (v3 - v1)
    cross_product = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])
    # If the cross product is positive, the vertices are already in counterclockwise order
    if cross_product > 0:
        return [v1, v2, v3]
    # If the cross product is negative, swap the second and third vertices to put them in counterclockwise order
    elif cross_product < 0:
        return [v1, v3, v2]
    # If the cross product is zero, the points are collinear, so we need to check their relative positions
    else:
        if v1[0] == v2[0] and v2[0] == v3[0]:
            # All points have the same x-coordinate, so sort by y-coordinate
            return sorted([v1, v2, v3], key=lambda v: v[1])
        elif v1[0] == v2[0]:
            # v1 and v2 have the same x-coordinate, so they're either above or below v3
            if v1[1] < v2[1]:
                return [v1, v2, v3]
            else:
                return [v2, v1, v3]
        elif v2[0] == v3[0]:
            # v2 and v3 have the same x-coordinate, so they're either above or below v1
            if v2[1] < v3[1]:
                return [v1, v2, v3]
            else:
                return [v1, v3, v2]
        elif v1[0] == v3[0]:
            # v1 and v3 have the same x-coordinate, so they're either above or below v2
            if v1[1] < v3[1]:
                return [v1, v3, v2]
            else:
                return [v3, v1, v2]
        else:
            # General case - calculate the slope of each side and sort by slope
            slopes = [(v2[1] - v1[1]) / (v2[0] - v1[0]), (v3[1] - v2[1]) / (v3[0] - v2[0]), (v1[1] - v3[1]) / (v1[0] - v3[0])]
            sorted_vertices = [v1, v2, v3]
            sorted_vertices.sort(key=lambda v, slopes=slopes: slopes[sorted_vertices.index(v)])
            return sorted_vertices

#checking whether the point lies in the triangle
def point_in_triangle(point, vertices):
    x3, y3 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    x, y = point
    # Compute the area of the triangle
    area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    # Compute the areas of the three sub-triangles
    sub_area1 = 0.5 * abs(x*(y2-y3) + x2*(y3-y) + x3*(y-y2))
    sub_area2 = 0.5 * abs(x1*(y-y3) + x*(y3-y1) + x3*(y1-y))
    sub_area3 = 0.5 * abs(x1*(y2-y) + x2*(y-y1) + x*(y1-y2))
    # Check if the sum of the sub-triangle areas equals the triangle area
    if abs(area - (sub_area1 + sub_area2 + sub_area3)) < 1e-6:
        return True
    else:
        return False
