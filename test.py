import torch

class Testing:
    def __init__(self, model, data, labels, ranking, iterations, size, device, diversity=False):
        self.model = model
        self.ranking = ranking
        self.cur_it = 0
        self.iterations = iterations
        self.accuracy = []
        self.size = size
        self.device = device
        self.labels = labels
        self.diversity = diversity
        self.trans_data = [[data]]
        acc = self.get_accuracy(data, labels)
        self.accuracy.append([acc])
    
    def single_testing(self, x, y, deltas):
        if self.diversity:
            _, picked_deltas = self.ranking(x, self.trans_data[0][0], deltas, topk=self.size)
        else:
            _, picked_deltas = self.ranking(x, y, deltas, topk=self.size)
        accuracy = []
        trans_data = []
        for i in range(len(picked_deltas)):
            x_trans = x+picked_deltas[i]
            trans_data.append(x_trans)
            acc = self.get_accuracy(x_trans, y)
            accuracy.append(acc)
        return trans_data, accuracy

    def testing(self, data_gen, l2_rad, linfty_rad, forward=True):
        while self.cur_it < self.iterations:
            curr_data = self.trans_data[self.cur_it]
            curr_trans_data = []
            curr_acc = []
            for x in curr_data:
                if forward:
                    deltas = data_gen(x)
                else:
                    deltas = data_gen(x, self.labels, l2_rad, linfty_rad)
                trans_data, accuracy = self.single_testing(x, self.labels, deltas)
                curr_acc += accuracy
                curr_trans_data += trans_data
            self.trans_data.append(curr_trans_data)
            self.accuracy.append(curr_acc)
            self.cur_it += 1

    def get_accuracy(self, x, y):
        self.model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            x = x.to(self.device)
            y = y.to(self.device)
            outputs = self.model.forward(x)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += y.size(0)
            n_correct += (predicted == y).sum().item()

            acc = 100.0 * n_correct / n_samples
        print(f'Accuracy is {acc}%')
        return acc