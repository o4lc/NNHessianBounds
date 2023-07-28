from fileinput import filename

# from nbformat import write
from packages import *

class Plotter():
    def __init__(self, plotCounter = 0, savingDirectory='./Output/'):
        self.plotCounter = plotCounter
        self.savingDirectory = savingDirectory
        self.images = []
        fileNames = next(os.walk(self.savingDirectory), (None, None, []))[2]

        fig = plt.figure()
        self.ax = fig.add_subplot(111)

        # for fileName in fileNames:
        #     os.remove(self.savingDirectory + fileName)

    def plotSpace(self, space, lowCoord, upperCoord):
        plt.rcParams["figure.figsize"] = [7.00, 7.00]
        plt.rcParams["figure.autolayout"] = True
        self.ax.set_aspect('equal', adjustable='box')

        rectangle = patches.Rectangle((lowCoord[0], lowCoord[1]), upperCoord[0] - lowCoord[0], upperCoord[1] - lowCoord[1], 
                                    edgecolor=None, facecolor="lightgrey", linewidth=7, alpha=0.5)
        self.ax.add_patch(rectangle)
        # print(space)
        for node in space:
            # rectangle = patches.Rectangle((node.coordLower[0], node.coordLower[1]), 
            #                             node.coordUpper[0] - node.coordLower[0], 
            #                             node.coordUpper[1] - node.coordLower[1], 
            #                             edgecolor=None, facecolor="green", linewidth=7, alpha=0.3)
            # self.ax.add_patch(rectangle)
            self.ax.vlines(x=node.coordLower[0], ymin=node.coordLower[1], 
                                ymax=node.coordUpper[1], color='black', linestyle='--', alpha=0.5, linewidth=2)
            
            self.ax.vlines(x=node.coordUpper[0], ymin=node.coordLower[1], 
                                ymax=node.coordUpper[1], color='black', linestyle='--', alpha=0.5, linewidth=2)
            
            self.ax.hlines(y=node.coordLower[1], xmin=node.coordLower[0], 
                                xmax=node.coordUpper[0], color='black', linestyle='--', alpha=0.5, linewidth=2)
            
            self.ax.hlines(y=node.coordUpper[1], xmin=node.coordLower[0], 
                                xmax=node.coordUpper[0], color='black', linestyle='--', alpha=0.5, linewidth=2)
        plt.xlim([lowCoord[0] - 0.2, upperCoord[0] + 0.2])
        plt.ylim([lowCoord[1] - 0.2, upperCoord[1] + 0.2])

        # plt.savefig(self.savingDirectory + str(self.plotCounter))

    def showAnimation(self, space, currDim):
        # Closing any previous open plots
        # for j in range(len(plt.get_fignums())):
        #     plt.close()

        # Creating the gif with the saved pictures
        # fig, ax = plt.subplots()
        # ax.set_aspect('equal', adjustable='box')

        # num = len(next(os.walk(self.savingDirectory), (None, None, []))[2])
        # self.images = []
        # for i in range(num):
        #     image = plt.imread(str(self.savingDirectory + str(i) + '.png'))
        #     im = ax.imshow(image, animated=True)
        #     self.images.append([im])
        # image = plt.imread(str(self.savingDirectory + space[0].scoreFunction + '.png'))
        # ani = animation.ArtistAnimation(fig, self.images, interval=3000, blit=True,
        #                         repeat_delay=2000)
        
        # # Clearing the Directory
        # fileNames = next(os.walk(self.savingDirectory), (None, None, []))[2]
        # for fileName in fileNames:
        #     os.remove(self.savingDirectory + fileName)

        # # plt.axis('off')
        # ax = plt.gca()
        # ax.axes.xaxis.set_ticklabels([])
        # plt.title('Branch and Bound Sequence')
        plt.xlabel('$x_0$')
        plt.ylabel('$x_1$')
        print('aa')
        # writergif = animation.PillowWriter(fps=2) 
        # ani.save(self.savingDirectory + 'Branch&Bound.gif', writer=writergif)

        # ax.imshow(image)
        plt.savefig(self.savingDirectory + str(currDim))
        plt.close()

        # plt.show()

        # plt.show()


def plotReachability(configFileToLoad, pcaDirections, indexToStartReadingBoundsForPlotting, 
                     calculatedLowerBoundsforpcaDirections, Method = 'secondOrder', finalIter=False, finalHorizon=1):
    ax = plt.gca()
    fig = plt.gcf()
    if True:
        AA = -np.array(pcaDirections[indexToStartReadingBoundsForPlotting:])
        AA = AA[:, :2]
        bb = []
        for i in range(indexToStartReadingBoundsForPlotting, len(calculatedLowerBoundsforpcaDirections)):
            bb.append(-calculatedLowerBoundsforpcaDirections[i])

        # print(AA, torch.vstack(bb))
        bb = np.array(bb)
        pltp = polytope.Polytope(AA, bb)
        if Method == 'secondOrder':
            ax = pltp.plot(ax, alpha = 0.3, color='grey', edgecolor='black')
        else:  
            ax = pltp.plot(ax, alpha = 0.5, color='None', edgecolor='blue')

        if finalIter:
            plt.axis("equal")
            if "robotarm" not in configFileToLoad.lower() and finalHorizon < 11:
                leg1 = plt.legend()
            # plot constraints
            if 'quad' in configFileToLoad.lower():
                e1 = patches.Ellipse((2, 4), 2, 2, color ='red', alpha = 0.25, hatch='//')
                e2 = patches.Ellipse((4, 2), 2, 2, color ='red', alpha = 0.25, hatch='//')
                ax.add_patch(e1)
                ax.add_patch(e2)
            plt.xlabel('$x_0$')
            plt.ylabel('$x_1$')
            custom_lines = [Line2D([0], [0], color='b', lw=2, linestyle='--'),
                        Line2D([0], [0], color='grey', lw=2, linestyle='--')]
            
            if 'quad' in configFileToLoad.lower() and True:
                ax.legend(custom_lines, ['ReachLipBnB', 'Our method'], loc=4)
                plt.gca().add_artist(leg1)

def SOTAPlotCMP(filename, verisig=True, numHorizons=10):
    ax = plt.gca()
    tmp = './Utilities/' + filename + '.txt'
    with open(tmp) as f:
        lines = f.readlines()

    if not verisig:
        if filename in ['B1', 'B3', 'B4']:
            for i, line in enumerate(lines):
                if i < 2 * numHorizons:
                    tmp = line.split("#")
                    if tmp[0] != '\n':
                        rectangle = patches.Rectangle(((float(tmp[1])), float(tmp[2])), float(tmp[3]), float(tmp[4]),
                                                        edgecolor='blue', facecolor="None", linewidth=2, alpha=0.3)
                        ax.add_patch(rectangle)
        
        custom_lines = [Line2D([0], [0], color='b', lw=2, linestyle='--'),
        Line2D([0], [0], color='grey', lw=2, linestyle='--')]
        ax.legend(custom_lines, ['Poly. Zono.', 'Ours'], loc=4)
        # plt.gca().add_artist(leg1)
    else:
        if filename in ['TORA', 'B2', 'B3', 'B5']:
            for line in lines:
                tmp = line.split(' ')
                cleaned = []
                for i in range(len(tmp)):
                    # print(tmp[i])
                    if tmp[i] in ['hold'] or 'clear' in tmp[i]:
                        break
                    if tmp[i] in [',', "'color'", '[', ']', 'plot(', "'[0", '0.4', "0]');\n"]:
                        pass
                    else:
                        cleaned.append((float(tmp[i])))
                else:
                    plt.plot(cleaned[:9], cleaned[9:], '--', c='blue', alpha=0.2)
        else:
            points = []
            for line in lines:
                if line != '\n':
                    tmp = line.split(' ')
                    points.append([float(tmp[0]), float(tmp[1][:-2])])
                else:
                    if len(points) > 1:
                        points = np.array(points)
                        plt.plot(points[:, 0], points[:, 1], '--', c='blue', alpha=0.2)
                        points = []
        
        
        # leg1 = plt.legend()
        custom_lines = [Line2D([0], [0], color='b', lw=2, linestyle='--'),
                Line2D([0], [0], color='grey', lw=2, linestyle='--')]
        ax.legend(custom_lines, ['Verisig2', 'Ours'], loc=4)
        # plt.gca().add_artist(leg1)
            
