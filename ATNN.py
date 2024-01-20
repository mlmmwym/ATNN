import numpy as np
import random
import json

def create_random_matrix(m, n):
    limit = (6 / (m + n)) ** 0.5
    return np.random.uniform(-limit, limit, (m,n))

def standardization(arr):
    mean = arr.mean()
    std = arr.std()
    arr = (arr-mean) / (std + 0.0000001)
    return arr

def matrix_sim(arr1,arr2):
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()

    mean1 = np.mean(farr1)
    mean2 = np.mean(farr2)

    numer = np.sum((farr1) * (farr2))
    denom = np.sqrt(np.sum((farr1)**2) * np.sum((farr2)**2))
    similar = numer / denom
    return similar

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def cross_entropy(y_hat, y):
    return -np.sum(y*np.log(y_hat))

def is_max_value_index_equal(V1, V2):
    if np.argmax(V1) == np.argmax(V2):
        return 1
    else:
        return 0

def save_data(data, file):
    with open(file, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class NODE:
    def __init__(self, hNeuronNum, cNeuronNum, branchType):
        self.branchType = branchType
        self.isShare = 0
        self.learnRate = 0.02
        self.hNeuronNum = hNeuronNum
        self.cNeuronNum = cNeuronNum
        self.isRootNode = 0
        self.childList = []
        self.parent = None
        self.hW = None
        self.hb = None
        self.cW = None
        self.cb = None
        self.hideInput = None
        self.hideOutput = None
        self.classifierInput = None
        self.classifierOutput = None
        self.d_hideInput = None
        self.d_classifierInput = None
        self.dev_hW = None
        self.dev_cW = None
        self.weight = 0
        self.depth = 0
        self.trainTimes = 0
        self.reduction = 0.999
        self.minLR = 0.001
        self.lastPredictLoss = 0
        self.squareGrad_hW = []
        self.squareGrad_hb = []
        self.Fisher_hW = []
        self.Fisher_hb = []
        self.alertSquareGrad_hW = []
        self.alertSquareGrad_hb = []
        self.alertFisher_hW = []
        self.alertFisher_hb = []
        self.lastConcept_hW = []
        self.lastConcept_hb = []
        self.init_b()

    def relu(self, x):
        output = standardization(np.maximum(x,0))
        return output

    def dev_relu(self, x):
        x[x>0] = 1
        x[x<0] = 0
        return x

    def init_weight(self):
        if self.isRootNode:
            self.hW = create_random_matrix(self.hNeuronNum, self.hNeuronNum)
            self.cW = create_random_matrix(self.cNeuronNum, self.hNeuronNum)
            self.squareGrad_hW = np.zeros((self.hNeuronNum, self.hNeuronNum))
            self.Fisher_hW = np.zeros((self.hNeuronNum, self.hNeuronNum))
            self.alertSquareGrad_hW = np.zeros((self.hNeuronNum, self.hNeuronNum))
            self.alertFisher_hW = np.zeros((self.hNeuronNum, self.hNeuronNum))
            self.lastConcept_hW = np.zeros((self.hNeuronNum, self.hNeuronNum))
            # pass
        else:
            self.hW = create_random_matrix(self.hNeuronNum, self.parent.hNeuronNum)
            self.cW = create_random_matrix(self.cNeuronNum, self.hNeuronNum)
            self.squareGrad_hW = np.zeros((self.hNeuronNum, self.parent.hNeuronNum))
            self.Fisher_hW = np.zeros((self.hNeuronNum, self.parent.hNeuronNum))
            self.alertSquareGrad_hW = np.zeros((self.hNeuronNum, self.parent.hNeuronNum))
            self.alertFisher_hW = np.zeros((self.hNeuronNum, self.parent.hNeuronNum))
            self.lastConcept_hW = np.zeros((self.hNeuronNum, self.parent.hNeuronNum))

    def init_b(self):
        self.hb = np.array([[0] * self.hNeuronNum]).T
        self.cb = np.array([[0] * self.cNeuronNum]).T
        self.Fisher_hb = np.array([[0] * self.hNeuronNum]).T
        self.squareGrad_hb = np.array([[0] * self.hNeuronNum]).T
        self.alertSquareGrad_hb = np.array([[0] * self.hNeuronNum]).T
        self.alertFisher_hb = np.array([[0] * self.hNeuronNum]).T
        self.lastConcept_hb = np.array([[0] * self.hNeuronNum]).T

class Model:
    def __init__(self, featureNum, hNeuronNum, cNeuronNum):
        self.featureNum = featureNum
        self.hNeuronNum = hNeuronNum
        self.cNeuronNum = cNeuronNum
        self.beta = 0.99
        self.smooth = 0.2
        self.trainTimes = 0
        self.activeBranch = 0
        self.branchNum = 0
        self.maxBranchNum = 20
        self.model = None
        self.nodeList = {}
        self.activeNodeList = []
        self.lossList = {}
        self.lossLen = 1000
        self.splitLen = 50
        self.lossStatisticsList = {}
        self.branchList = []
        self.driftAlert = 0
        self.alertNum = 0
        self.lastDriftTime = 0
        self.lamda = 5000
        self.dataSet = None
        self.confid = 3

        self.init_model()

    def init_model(self):
        branchType = 0
        root = NODE(self.featureNum,self.cNeuronNum,branchType)
        root.isRootNode = 1
        root.init_weight()
        self.nodeList[0] = []
        self.nodeList[0].append(root)
        self.model = root
        parent = root
        for i in range(2):
            self.add_child_node(parent,branchType)
            parent = parent.childList[-1]
        self.activeBranch = branchType
        self.init_node_weight()

    def weight_sim(self):
        branchNode = self.model
        init_sim = 0.85
        max_sim = 0.85
        min_sim = 0.7
        depth = 0
        weightList = [node.weight for node in self.nodeList[0]]
        max_index = 0 
        for i in range(len(weightList)):
            if weightList[i] > sum(weightList)/len(weightList):
                max_index = i
        weightList = []
        for node in self.nodeList[0][1:]:
            node.alertFisher_hW = node.alertSquareGrad_hW / self.alertNum
            node.alertFisher_hb = node.alertSquareGrad_hb / self.alertNum
            sim = matrix_sim(node.alertFisher_hW, node.Fisher_hW)
            weightList.append([sim, node.weight])
            if node.depth <= max_index:
                if sim < init_sim:
                    branchNode = node
                    init_sim = sim
                if sim < min_sim:
                    branchNode = node
        parent = branchNode.parent
        branchNode.isShare = 1
        while parent:
            parent.isShare = 1
            parent = parent.parent
        return branchNode


    def add_branch(self):
        parent = self.weight_sim()
        self.branchNum = self.branchNum + 1
        self.branchList.append(self.branchNum)
        self.activeBranch = self.branchNum
        for i in range(1):
            self.add_child_node(parent,self.branchNum)
            parent = parent.childList[-1]
        if len(self.branchList) > self.maxBranchNum:
            delBranch = self.branchList[0]
            self.branchList = self.branchList[1:]
            self.del_branch(self.model,delBranch)
            del self.lossStatisticsList[delBranch]
            del self.lossList[delBranch]
            del self.nodeList[delBranch]

    def add_child_node(self,parentNode,branchType,weight = None):
        child = NODE(self.hNeuronNum,self.cNeuronNum,branchType)
        child.parent = parentNode
        child.depth = child.parent.depth + 1
        child.init_weight()
        if weight:
            child.weight = weight
        parentNode.childList.append(child)
        if branchType not in self.nodeList.keys():
            self.nodeList[branchType] = []
        self.nodeList[branchType].append(child)

    def forward_propagation(self, node, feature):
        if node.isRootNode:
            node.hideInput = feature
            node.hideOutput = feature
            node.classifierInput = np.dot(node.cW, node.hideOutput) + node.cb
            node.classifierOutput = softmax(node.classifierInput)
        else:
            node.hideInput = np.dot(node.hW, feature) + node.hb
            node.hideOutput = node.relu(node.hideInput)
        
            node.classifierInput = np.dot(node.cW, node.hideOutput) + node.cb
            node.classifierOutput = softmax(node.classifierInput)

        for child in node.childList:
            if not self.driftAlert and child.branchType != self.activeBranch and child.branchType != 0:
                continue
            self.forward_propagation(child,node.hideOutput)

    def back_propagation(self, node, trueLabel):
        for child in node.childList:
            if child.branchType == 0 or child.branchType == self.activeBranch:
                self.back_propagation(child,trueLabel)
        node.lastPredictLoss = cross_entropy(node.classifierOutput, trueLabel)
        node.dev_cInput = node.weight * (node.classifierOutput - trueLabel)
        node.dev_cW = np.dot(node.dev_cInput, node.hideOutput.T)
        if node.isRootNode: 
            return
        node.dev_hInput = np.dot(node.cW.T, node.dev_cInput) * node.dev_relu(node.hideInput)
        node.dev_hW = np.dot(node.dev_hInput, node.parent.hideOutput.T)

        if self.driftAlert and node.branchType == 0:
            node.alertSquareGrad_hW = node.alertSquareGrad_hW + np.maximum(node.dev_hW,-node.dev_hW)
            node.alertSquareGrad_hb = node.alertSquareGrad_hb + node.dev_hInput

        if node.branchType != self.activeBranch:
            node.dev_cW = node.dev_cW * 0
            node.dev_hInput = node.dev_hInput * 0
            node.dev_hW = node.dev_hW * 0

        for child in node.childList:
            if child.branchType != 0 and child.branchType != self.activeBranch:
                continue
            child_dev_hInput = np.dot(child.hW.T, child.dev_hInput) * node.dev_relu(node.hideInput)
            node.dev_hInput = node.dev_hInput + child_dev_hInput
        node.dev_hW = np.dot(node.dev_hInput, node.parent.hideOutput.T)

        if node.branchType == 0 and node.trainTimes > 2000:
            node.squareGrad_hW = node.squareGrad_hW + (node.dev_hW + self.lamda * node.Fisher_hW * (node.hW - node.lastConcept_hW)) ** 2
            node.squareGrad_hb = node.squareGrad_hb + (node.dev_hInput + self.lamda * node.Fisher_hb * (node.hb - node.lastConcept_hb)) ** 2


    def update_model(self, node):
        for child in node.childList:
            if child.branchType == 0 or child.branchType == self.activeBranch:
                self.update_model(child)
        node.trainTimes = node.trainTimes + 1
        lr = node.learnRate
        if node.isRootNode: 
            node.cW = node.cW - node.dev_cW * lr
            node.cb = node.cb - node.dev_cInput * lr
        else:
            if node.isShare: 
                node.hW = node.hW - (node.dev_hW + self.lamda * node.Fisher_hW * (node.hW - node.lastConcept_hW)) * lr
                node.cW = node.cW - node.dev_cW * lr
                node.hb = node.hb - (node.dev_hInput + self.lamda * node.Fisher_hb * (node.hb - node.lastConcept_hb)) * lr
                node.cb = node.cb - node.dev_cInput * lr
            else:
                node.hW = node.hW - node.dev_hW * lr
                node.cW = node.cW - node.dev_cW * lr
                node.hb = node.hb - node.dev_hInput * lr
                node.cb = node.cb - node.dev_cInput * lr

        if node.learnRate > node.minLR:
            node.learnRate = node.learnRate * node.reduction

        node.dev_hW = None
        node.dev_cW = None
        node.dev_hInput = None
        node.dev_cInput = None


    def del_branch(self,node,branchType):
        for child in node.childList:
            self.del_branch(child,branchType)
        if node.branchType == branchType:
            node.parent.childList.remove(node)

    def get_active_node_list(self):
        return self.nodeList[self.activeBranch]

    def init_node_weight(self):
        activeNodeList = self.get_active_node_list()
        avgWeight = 1.0/len(activeNodeList)
        for node in activeNodeList:
            node.weight = avgWeight

    def get_model_dict_structure(self,node):
        nodeName = str(node.branchType) + '-' + '%.5f' % node.weight  + "-" + str(node.isShare)
        if len(node.childList) == 0:
            return nodeName
        else:
            tree_dict = {nodeName: {}}
            for i in range(len(node.childList)):
                tree_dict[nodeName]["%s" % i] = self.get_model_dict_structure(node.childList[i])
            return tree_dict

    def get_model_output(self):
        modelOutput = {}
        for branch,nodeList in self.nodeList.items():
            branchOutput = sum([node.classifierOutput * node.weight for node in nodeList])
            modelOutput[branch] = branchOutput / sum(branchOutput)
        return modelOutput

    def update_branch_predict_loss(self,modelOutput,label):
        for branch,branchOutput in modelOutput.items():
            if branch not in self.lossList.keys():
                self.lossList[branch] = []
            self.lossList[branch].append(cross_entropy(branchOutput, label))
            if len(self.lossList[branch]) > self.lossLen:
                self.lossList[branch] = self.lossList[branch][-self.lossLen:]

    def update_loss_statistics(self):
        for branch,lossList in self.lossList.items():
            if len(lossList) < self.lossLen:
                continue
            mean = np.mean(lossList)
            var = np.std(lossList)
            prev_loss = sorted(lossList[-self.splitLen:])[0:-5]
            prev_mean = np.mean(prev_loss)
            prev_var = np.std(prev_loss)
            if branch not in self.lossStatisticsList.keys():
                self.lossStatisticsList[branch] = {
                    "mean":mean,
                    "var":var,
                    "prev_mean":prev_mean,
                    "prev_var":prev_var,
                }
            else:
                if mean + var < self.lossStatisticsList[branch]["mean"] + self.lossStatisticsList[branch]["var"]:
                    self.lossStatisticsList[branch]["mean"] = mean
                    self.lossStatisticsList[branch]["var"] = var

                self.lossStatisticsList[branch]["prev_mean"] = prev_mean
                self.lossStatisticsList[branch]["prev_var"] = prev_var

        if self.activeBranch in self.lossStatisticsList.keys():
            d = self.lossStatisticsList[self.activeBranch]["mean"] + self.confid * self.lossStatisticsList[self.activeBranch]["var"]
            lossWin = self.lossList[self.activeBranch][-5:]
            driftFlag = min([l - d for l in lossWin])
            if not self.driftAlert:
                if driftFlag > 0:
                    self.driftAlert = 1
                else:
                    lossWin = sorted(self.lossList[self.activeBranch][-self.splitLen:])
                    mean = np.mean(lossWin[0:-5])
                    var = np.var(lossWin[0:-5])
                    if mean + var > d:
                        self.driftAlert = 1
            if self.driftAlert:
                self.alertNum = self.alertNum + 1

    def reset_weight(self):
        activeNodeList = self.get_active_node_list()
        for node in activeNodeList:
            node.weight = 1.0/len(activeNodeList)

    def concept_detection(self):
        if self.activeBranch not in self.lossStatisticsList.keys():
            return
        activeBranchLoss = self.lossStatisticsList[self.activeBranch]
        minLossBranch = min(self.lossStatisticsList.keys(), key = lambda k:self.lossStatisticsList[k]["prev_mean"])
        branchLoss = self.lossStatisticsList[minLossBranch]
        if branchLoss["prev_var"] + branchLoss["prev_mean"] > branchLoss["mean"] + self.confid * branchLoss["var"]:
            self.update_fisherMatrix()
            self.add_branch()
            self.trainType = 0
            self.lastDriftTime = self.trainTimes
            self.reset_weight()
        else:
            if minLossBranch != self.activeBranch:
                self.activeBranch = minLossBranch
            else:
                pass
        self.driftAlert = 0
        self.alertNum = 0
        for node in self.nodeList[0]:
            node.alertSquareGrad_hW = node.alertSquareGrad_hW*0
            node.alertSquareGrad_hb = node.alertSquareGrad_hb*0

    def update_fisherMatrix(self):
        for node in self.nodeList[0]:
            node.lastConcept_hb = node.hb
            node.lastConcept_hW = node.hW
            node.Fisher_hW = node.squareGrad_hW / node.trainTimes
            node.Fisher_hb = node.squareGrad_hb / node.trainTimes


    def train_model(self, feature, label):
        model = self.model
        self.forward_propagation(model, feature)
        modelOutput = self.get_model_output()
        result = modelOutput[self.activeBranch]
        self.update_branch_predict_loss(modelOutput,label)
        self.update_loss_statistics()
        self.back_propagation(model, label)
        self.update_model(model)
        activeNodeList = self.get_active_node_list()
        self.update_weight_by_loss(activeNodeList, label)
        self.trainTimes = self.trainTimes + 1
        self.model_grow_and_prune()
        if self.alertNum == self.splitLen:
            self.concept_detection()
        return result

    def model_grow_and_prune(self):
        activeNodeList = self.get_active_node_list()
        wlist = [node.weight for node in activeNodeList]
        maxWeightNode = activeNodeList[wlist.index(max(wlist))]
        if len(maxWeightNode.childList) < 1:
            self.add_child_node(maxWeightNode,maxWeightNode.branchType, weight = 1.0/ (len(activeNodeList) + 1))

    def update_weight_by_loss(self, nodeList, label):
        zt = 0
        losses = []
        for node in nodeList:
            losses.append(node.lastPredictLoss)
        M = sum(losses)
        losses = [loss / M for loss in losses]
        min_loss = min(losses)
        max_loss = max(losses)
        range_of_loss = (max_loss - min_loss) + 0.000001
        losses = [(loss - min_loss) / range_of_loss for loss in losses]

        for i in range(len(nodeList)):
            newWeight = nodeList[i].weight * (self.beta**losses[i])
            if newWeight < self.smooth / len(nodeList):
                newWeight = self.smooth / len(nodeList)
            nodeList[i].weight = newWeight
            zt = zt + newWeight
        for node in nodeList:
            node.weight = node.weight / zt


def onlineLearning(dataSet):
    dataSetName = dataSet
    fileName = "D:/dataset/" + dataSetName + ".npz"
    data = np.load(fileName)
    x_train = data["x_train"]
    y_train = data["y_train"]
    statisticsLen = 100
    featureLen = len(x_train[0])
    hNeuronNum = 256
    classNum = len(y_train[0])
    model = Model(featureLen, hNeuronNum, classNum)
    model.init_node_weight()
    predictRightNumber = 0
    exampleNumber = 0
    resultList = {}
    blockResultList = []
    for i in range(len(x_train)):
        exampleNumber = exampleNumber + 1
        feature = np.array([x_train[i]]).T
        label = np.array([y_train[i]]).T
        predictResult = model.train_model(feature, label)
        blockResultList.append(is_max_value_index_equal(predictResult, label))
        if len(blockResultList)==statisticsLen or exampleNumber == len(x_train):
            predictRightRate = sum(blockResultList) / len(blockResultList)
            predictRightNumber = predictRightNumber + sum(blockResultList)
            print("*************************************************************************")
            print("size: ", exampleNumber)
            print("realtime: ",predictRightRate)
            print("cumulative: ", predictRightNumber / exampleNumber)
            resultList[exampleNumber] = {"realtime":predictRightRate,"cumulative":predictRightNumber / exampleNumber}
            blockResultList = []
    save_data(resultList, dataSet + ".json")


def main():
    dataSetList = ["RBF2_0"]
    for dataSet in dataSetList:
        print("-----------------------------")
        print("start train : ",dataSet)
        onlineLearning(dataSet)
main()