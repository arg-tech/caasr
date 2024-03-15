from collections import defaultdict
from itertools import combinations

class ArgumentStructureGenerator:
    def __init__(self):
        pass
    
    def find_all_paths(self, graph, start, end, path=None):
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = self.find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def construct_tree(self, sentences, relations):
        graph = defaultdict(list)
        for parent, child, relation_type in relations:
            graph[parent].append(child)
        all_paths = {}
        for start, end in combinations(sentences, 2):
            paths = self.find_all_paths(graph, start, end)
            if paths:
                all_paths[(start, end)] = paths
        used_relations = set()
        for paths in all_paths.values():
            for path in paths:
                for i in range(len(path) - 1):
                    used_relations.add((path[i], path[i+1]))
        tree = {}
        for parent, child, relation_type in relations:
            if (parent, child) in used_relations:
                if parent not in tree:
                    tree[parent] = []
                tree[parent].append((child, relation_type))
        for start, end in all_paths:
            if len(all_paths[(start, end)]) > 1:
                for i in range(len(all_paths[(start, end)]) - 1):
                    path1 = all_paths[(start, end)][i]
                    path2 = all_paths[(start, end)][i+1]
                    min_length = min(len(path1), len(path2))
                    for j in range(min_length):
                        parent = path1[j]
                        child = path2[j]
                        if (parent, child) in tree[start]:
                            tree[start].remove((parent, child))
        return tree

    def get_paths(self, tree, start, end, path=None):
        if path is None:
            path = []
        path.append(start)
        if start == end:
            return [path]
        if start not in tree:
            return []
        paths = []
        for child, relation_type in tree[start]:
            if child not in path:            
                newpaths = self.get_paths(tree, child, end, path[:])
                for newpath in newpaths:
                    newpath.append(relation_type)
                    paths.append(newpath)
        return paths

    def generate_argument_structure_from_relations(self, sentences, relations):
        argument_relations = {}
        tree = self.construct_tree(sentences, relations)
        all_paths = []
        for i in range(len(sentences) - 1):
            start = sentences[i]
            end = sentences[i + 1]
            paths = self.get_paths(tree, start, end)
            all_paths.extend(paths)
        for path in all_paths:
            argument_relations[path[0]] = (path[1:])
        return argument_relations


# Example usage
generator = ArgumentStructureGenerator()
sentences = ['A', 'B', 'C', 'D']
relations = [('A', 'B', 'support'), ('A', 'C', 'attack'), ('B', 'D', 'support'), ('C', 'D', 'support'), ('B', 'C', 'support')]
generator.generate_argument_structure_from_relations(sentences, relations)
