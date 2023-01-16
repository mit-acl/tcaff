import numpy as np

class InconsistencyCounter():
    
    def __init__(self):
        self.reset()
        self.verbose = False
        
    def reset(self):
        self.associations = dict()
        self.elements = list()
    
    def add_groups(self, agent_num, groups):
        for group in groups:
            self.add_group(agent_num, group)
        
    def add_group(self, agent_num, group):
        agent_elements = []
        non_agent_elements = []
        for el in group:
            agent_id, el_id = el
            if agent_id == agent_num:
                agent_elements.append(el)
                assert el not in self.elements, f'element should only be added once.\nagent_num: {agent_num}\nself.associations: {self.associations}\nself.elements: {self.elements}\nel: {el}\ngroup: {group}'
                self.elements.append(el)
            else:
                non_agent_elements.append(el)
        for agent_el in agent_elements:
            if agent_el not in self.associations:
                self.associations[agent_el] = list()
            for other_el in non_agent_elements:
                self.associations[agent_el].append(other_el)
                
    def count_inconsistencies(self):
        if not self.elements:
            return 0
        to_add = set()
        for _, associations in self.associations.items():
            for a in associations:
                if a not in self.elements:
                    self.elements.append(a)
                if a not in self.associations:
                    to_add.add(a)
        for a in to_add:
            self.associations[a] = []
        # construct association matrix Y
        self.elements = sorted(self.elements, key=lambda x: x[0])
        Y = np.eye(len(self.elements), dtype=np.int8)
        Y_new = np.eye(len(self.elements), dtype=np.int8)
        for el, associations in self.associations.items():
            el_idx = self.elements.index(el)
            for a in associations:
                a_idx = self.elements.index(a)
                Y_new[el_idx, a_idx] = 1
        if self.verbose:
            print(self.elements)
            print(self.associations)
            print(Y_new)
            print('')
        while not np.array_equal(Y, Y_new):
            Y = np.copy(Y_new)
            for i in range(Y.shape[0]):
                for j in range(Y.shape[0]):
                    if i == j: continue
                    Y_new[i,j] = Y_new[i,j] | Y[j,i]
                    Y_new[j,i] = Y[i,j] | Y_new[j,i]
                    if Y[i,j]:
                        for k in range(Y.shape[0]):
                            Y_new[i,k] = Y_new[i,k] | Y[j,k]
            if self.verbose:
                print(Y_new)
                print('')

        agent_block_idx = [0]
        for i in range(1, len(self.elements)):
            if self.elements[i][0] != self.elements[i-1][0]:
                agent_block_idx.append(i)
        agent_block_idx.append(len(self.elements))
        inconsistencies = 0
        for i in range(len(agent_block_idx) - 1):
            for j in range(agent_block_idx[i], agent_block_idx[i+1]):
                num_pairs = 0
                for k in range(j, agent_block_idx[i+1]):
                    if Y_new[j,k]: num_pairs += 1
                inconsistencies += num_pairs - 1
        if self.verbose:
            print(agent_block_idx)
            print(inconsistencies)
        self.reset()
        if inconsistencies:
            print(agent_block_idx)
            print(Y_new)
            print(inconsistencies)
        return inconsistencies
        
                
if __name__ == '__main__':
    # TEST
    ic = InconsistencyCounter()
    ic.add_group(0, {(0,0), (1,0)})
    ic.add_group(1, {(1,0)})
    ic.add_group(1, {(1,1)})
    ic.count_inconsistencies()

    ic = InconsistencyCounter()
    ic.add_group('A', {('A', 1), ('D', 1)})
    ic.add_group('A', {('A', 2), ('B', 1), ('D', 2)})
    ic.add_group('A', {('A', 3), ('B', 3)})
    ic.add_group('B', {('A', 2), ('B', 1), ('C', 1)})
    ic.add_group('B', {('B', 2), ('C', 2)})
    ic.add_group('B', {('A', 3), ('B', 3), ('C', 3)})
    ic.add_group('C', {('B', 1), ('C', 1), ('D', 1)})
    ic.add_group('C', {('B', 2), ('C', 2), ('D', 2)})
    ic.add_group('C', {('B', 3), ('C', 3)})
    ic.add_group('D', {('A', 1), ('C', 1), ('D', 1)})
    ic.add_group('D', {('A', 2), ('C', 2), ('D', 2)})
    ic.count_inconsistencies()