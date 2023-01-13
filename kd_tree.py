import math
import time


class Node():
    def __init__(self, lchild, rchild, value, split_dim):
        self.lchild = lchild  # left subtree of the node
        self.rchild = rchild  # right subtree of the node
        self.value = value  # node value
        self.split_dim = split_dim  # the dimension used to make the partition


class Rectangle():
    def __init__(self, lower_point, up_point):
        self.lower = lower_point
        self.upper = up_point

    def is_contains(self, point):
        return self.lower[0] <= point[0] <= self.upper[0] and self.lower[1] <= point[1] <= self.upper[1]


class KDTree():
    def __init__(self, data):
        self.dims = len(data[0])  # total characteristic number
        self.nearest_point = None
        self.nearest_dist = math.inf  # initialize to infinity

    def insert(self, current_data, split_dim):
        # Set a recursive exit: exit when all samples are divided
        if len(current_data) == 0:
            return None

        mid = self.cal_current_medium(current_data)  # calculate the index of the median
        data_sorted = sorted(current_data,
                             key=lambda x: x[split_dim])  # sort by segmentation dimension from smallest to largest

        # the following three sentences of code are essentially post-order traversals of binary trees
        lchild = self.insert(data_sorted[0:mid],
                             self.cal_split_dim(split_dim))  # recursively construct the left subtree
        rchild = self.insert(data_sorted[mid + 1:],
                             self.cal_split_dim(split_dim))  # recursively construct the right subtree
        return Node(lchild, rchild, data_sorted[mid],
                    split_dim)  # join the left and right subtrees from the root node and return

    # calculate the next partition dimension
    def cal_split_dim(self, split_dim):
        return (split_dim + 1) % self.dims

    # calculate the subscript of the median of the current dimension
    def cal_current_medium(self, current_data):
        return len(current_data) // 2

    # calculate the Euclidean distance between two points
    def cal_dist(self, sample1, sample2):
        return math.sqrt((sample1[0] - sample2[0]) ** 2 + (sample1[1] - sample2[1]) ** 2)

    # pass in the root node of kd tree and the point element to be searched, and search the nearest neighbor point of element
    def neighbor_search(self, node, element):
        if node is None:
            return
        # calculate the distance between the target node on the current partition dimension and the single dimension of the current node
        dist = node.value[node.split_dim] - element[node.split_dim]
        # search forward
        if dist > 0:  # the current node is above or to the left of the target node (in two dimensions)
            self.neighbor_search(node.lchild, element)  # recursively search the left subtree
        else:  # otherwise, the current node is below or to the right of the target node (in two dimensions)
            self.neighbor_search(node.rchild, element)  # recursively search the right subtree
        # calculate the Euclidean distance between the target node and the current node
        curr_dist = self.cal_dist(node.value, element)
        # Update the nearest neighbor node
        if curr_dist < self.nearest_dist:
            self.nearest_dist = curr_dist
            self.nearest_point = node
            # print(self.nearest_point.value)
        # backtrack Compare whether the nearest distance exceeds the distance between the target node and the current
        # node in the current dimension. If the distance exceeds the distance between the target node and the current
        # node in the current dimension, it indicates that there may be a closer point in the other subtree of the
        # current node, so it is necessary to search in the other subtree of the current node
        if self.nearest_dist > abs(dist):
            # since the search is done in the subtree on the other side of the current node, it is the exact opposite of the previous forward search
            if dist > 0:
                self.neighbor_search(node.rchild, element)
            else:
                self.neighbor_search(node.lchild, element)

    def get_nearest(self, root, element):
        self.neighbor_search(root, element)
        return self.nearest_point.value, self.nearest_dist

    def query(self, node, sample):
        '''
                determine whether there is a node in the binary tree whose value is equal to k
                :param k:
                :return:
                '''
        while node != None and node.value != sample:
            # calculate the distance between the target node on the current partition dimension and the single dimension of the current node
            dist = node.value[node.split_dim] - sample[node.split_dim]
            # search forward
            if dist > 0:  # the current node is above or to the left of the target node (in two dimensions)
                node = node.lchild
            else:  # otherwise, the current node is below or to the right of the target node (in two dimensions)
                node = node.rchild  # recursively search the right subtree

        return True if node != None else False

    def range(self, rectangle, root):
        l = []

        def ran(rec=Rectangle, node=Node, k=0):
            if not node:
                return
            axis = k % 2
            if axis == 0:
                if rec.lower[0] <= node.value[0] <= rec.upper[0] and rec.lower[1] <= node.value[1] <= rec.upper[1]:
                    l.append(node.value)
                if rec.lower[0] > node.value[0]:
                    ran(rec, node.rchild, k + 1)
                if rec.upper[0] < node.value[0]:
                    ran(rec, node.lchild, k + 1)
                if rec.lower[0] <= node.value[0] <= rec.upper[0]:
                    ran(rec, node.rchild, k + 1)
                    ran(rec, node.lchild, k + 1)

            else:
                if rec.lower[0] <= node.value[0] <= rec.upper[0] and rec.lower[1] <= node.value[1] <= rec.upper[1]:
                    l.append(node.value)
                if rec.lower[1] > node.value[1]:
                    ran(rec, node.rchild, k + 1)
                if rec.upper[1] < node.value[1]:
                    ran(rec, node.lchild, k + 1)
                if rec.lower[1] <= node.value[1] <= rec.upper[1]:
                    ran(rec, node.rchild, k + 1)
                    ran(rec, node.lchild, k + 1)

        ran(rectangle, root, k=0)
        return l


class Naive_method():
    def __init__(self, points):
        self.points = points

    def query(self, current_point):
        flag = False
        for point in self.points:
            if current_point == point:
                flag = True
                break
        return flag

    def range(self, rectangle):
        result1 = [p for p in self.points if rectangle.is_contains(p)]
        return result1


def performance_test():
    points = [[x, y] for x in range(1000) for y in range(1000)]
    kd = KDTree(points)
    root = kd.insert(points, 0)
    for i in range(500):
        lower = [i, i]
        upper = [1000-i, 1000-i]
        rectangle = Rectangle(lower, upper)

        #  naive method
        naive_method = Naive_method(points)
        start = int(round(time.time() * 1000))
        result1 = naive_method.range(rectangle)
        end = int(round(time.time() * 1000))
        print(f'Naive method: {end - start}ms')

        # k-d tree
        start = int(round(time.time() * 1000))
        result2 = kd.range(Rectangle([i, i], [1000-i, 1000-i]), root)
        end = int(round(time.time() * 1000))
        print(f'K-D tree: {end - start}ms')

    assert sorted(result1) == sorted(result2)


if __name__ == '__main__':
    performance_test()
