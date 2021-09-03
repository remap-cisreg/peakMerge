# %%
import numpy as np
from scipy.signal import argrelextrema, oaconvolve
from scipy.signal.windows import gaussian
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
import seaborn as sns
import pandas as pd
import argparse
import os
import sys




class peakMerger:
    """
    peakMerger

    Find consensus peaks from multiple peak calling files, and has methods
    to perform data exploration analyses.

    Parameters
    ----------
    genomeFile: string
        Path to tab separated (chromosome, length) annotation file.

    scoreMethod: "binary" or integer (optional, default "binary")
        If set to binary, will use a binary matrix (peak is present or absent).
        Otherwise if set to an integer this will be the column index (0-based)
        used as score. Usually not recommended but the code has fallbacks
        for the non-binary case.

    outputPath: string (optional, default None)
        Folder to export results (matrix, plots...). If not specified will not
        write any results.

    Attributes
    -----------------
    matrix: ndarray of shape (num consensuses, num experiments)
        The experiment-consensus peak presence matrix. It is
        a binary matrix unless the score method has been changed.

    consensuses: pandas DataFrame in bed-like format
        The genomic locations matching the rows (consensuses) of 
        the matrix attribute.
    
    labels: list of strings of size (num experiments)
        The file names matching the columns (experiments) of
        the matrix attribute.

    clustered: (ndarray of shape (num consensuses), ndarray of shape (num experiments))
        Two arrays (first are the consensuses, second are the experiments).
        Index of the cluster each sample belongs to.

    embedding: (ndarray of shape (num consensuses, 2), ndarray of shape (num experiments, 2))
        Two arrays (first are the consensuses, second are the experiments).
        Position of the points in the UMAP 2D space.

    """
    def __init__(self, genomeFile, outputPath=None, scoreMethod="binary"):
        self.score = scoreMethod
        self.outputPath = outputPath
        try:
            os.mkdir(self.outputPath)
        except:
            pass
        if not scoreMethod == "binary":
            self.score = int(scoreMethod)
        with open(genomeFile, 'r') as f:
            self.chrLen = dict()
            self.genomeSize = 0
            for l in f:
                l = l.rstrip("\n").split("\t")
                self.chrLen[l[0]] = int(l[1])
                self.genomeSize += int(l[1])    
        self.embedding = [None, None]
        self.clustered = [None, None]
        self.matrix = None
        self.consensuses = None
        self.labels = None

    def mergePeaks(self, folderPath, fileFormat, inferCenter=False, forceUnstranded=False, 
                  sigma="auto_1", perPeakDensity=False, perPeakMultiplier=0.5,
                  minOverlap=2):
        """
        Read peak called files, generate consensuses and the matrix.

        Parameters
        ----------
        folderPath: string
            Path to either folder with ONLY the peak files or comma separated
            list of files. Accepts compressed files.
        
        fileFormat: "bed" or "narrowPeak"
            Format of the files being read. 
            Bed file format assumes the max signal position to be at the 6th column 
            (0-based) in absolute coordinates.
            The narrowPeak format assumes the max signal position to be the 9th column 
            with this position being relative to the start position.

        inferCenter: boolean (optional, default False)
            If set to true will use the position halfway between start and end 
            positions. Enable this only if the summit position is missing.

        forceUnstranded: Boolean (optional, default False)
            If set to true, assumes all peaks are not strand-specific.

        sigma: float or "auto_1" or "auto_2" (optional, default "auto_1")
            Size of the gaussian filter (lower values = more separation).
            Only effective if perPeakDensity is set to False. 
            "auto_1" automatically selects the filter width at 1/8th of the average peak size.
            "auto_2" automatically selects the filter width based on the peak 
            count and genome size.
        
        perPeakDensity: Boolean (optional, default False)
            If set to false will perform a gaussian filter along the genome (faster),
            assuming all peaks have roughly the same size.
            If set to true will create the density curve per peak based on each peak
            individual size. This is much more slower than the filter method.
            May be useful if peaks are expected to have very different sizes. Can
            also be faster when the number of peaks is small.

        perPeakMultiplier: float (optional, default 0.25)
            Only effective if perPeakDensity is set to True. Adjusts the width of 
            the gaussian fitted to each peak (lower values = more separation).
        
        minOverlap: integer (optional, default 2)
            Minimum number of peaks required at a consensus.
        

        """
        alltabs = []
        if os.path.isdir(folderPath):
            files = os.listdir(folderPath)
        else:
            files = folderPath.split(",")
        for f in files:
            fmt = fileFormat
            if not fmt in ["bed", "narrowPeak"]:
                raise TypeError(f"Unknown file format : {fmt}")
            # Read bed format
            if fmt == "bed":
                if inferCenter:
                    usedCols = [0,1,2,5]
                else:
                    usedCols = [0,1,2,5,6]
                if not self.score == "binary":
                    usedCols.append(self.score)
                tab = pd.read_csv(folderPath + "/" + f, sep="\t", header=None, 
                                  usecols=usedCols)
                tab = tab[usedCols]
                if self.score == "binary":
                    tab[5000] = 1
                tab.columns = np.arange(len(tab.columns))
                tab[0] = tab[0].astype("str", copy=False)
                tab[3].fillna(value=".", inplace=True)
                if inferCenter:
                    tab[5] = tab[4]
                    tab[4] = ((tab[1]+tab[2])*0.5).astype(int)
                tab[6] = f
                if self.score == "binary":
                    tab[5] = [1]*len(tab)
                alltabs.append(tab)
            elif fmt == "narrowPeak":
                if inferCenter:
                    usedCols = [0,1,2,5]
                else:
                    usedCols = [0,1,2,5,9]
                if not self.score == "binary":
                    usedCols.append(self.score)
                tab = pd.read_csv(folderPath + "/" + f, sep="\t", header=None, 
                                  usecols=usedCols)
                tab = tab[usedCols]
                if self.score == "binary":
                    tab[5000] = 1
                tab.columns = np.arange(len(tab.columns))
                tab[0] = tab[0].astype("str", copy=False)
                tab[3].fillna(value=".", inplace=True)
                if inferCenter:
                    tab[5] = tab[4]
                    tab[4] = ((tab[1]+tab[2])*0.5).astype(int, copy=False)
                else:
                    tab[4] = (tab[1] + tab[4]).astype(int, copy=False)
                tab[6] = f
                alltabs.append(tab)
        # Concatenate files
        self.df = pd.concat(alltabs)
        # self.df[[0,1,2]].to_csv("h3k27ac_output/allPeaks.bed", sep="\t", header=False, index=False)
        self.numElements = len(self.df)
        self.avgPeakSize = np.mean(self.df[2] - self.df[1])
        # Check strandedness
        if forceUnstranded == True:
            self.df[3] = "."
            self.strandCount = 1
        else:
            # Check if there is only stranded or non-stranded elements
            strandValues = np.unique(self.df[3])
            self.strandCount = len(strandValues)
            if self.strandCount > 2:
                raise ValueError("More than two strand directions !")
            elif self.strandCount == 2 and "." in strandValues:
                raise ValueError("Unstranded and stranded values !")
        # Factorize experiments
        self.df[6], self.labels = pd.factorize(self.df[6])
        # Split per strand
        self.df = dict([(k, x) for k, x in self.df.groupby(3)])
        ########### Peak separation step ########### 
        # Compute sigma if automatic setting
        if sigma == "auto_1":   
            sigma = self.avgPeakSize/8.0
        elif sigma == "auto_2":
            l = self.genomeSize/self.numElements*self.strandCount
            sigma = np.log(10.0) * l / np.sqrt(2.0)
        else:
            sigma = float(sigma)
        if perPeakDensity:
            sigma = perPeakMultiplier
        windowSize = int(8*sigma)+1
        sepPerStrand = {}
        sepIdxPerStrand = {}
        # Iterate for each strand
        for s in self.df.keys():
            # Split peaks per chromosome
            posPerChr = dict([(k, x.values[:, [1,2,4]].astype(int)) for k, x in self.df[s].groupby(0)])
            # Iterate over all chromosomes
            sepPerStrand[s] = {}
            sepIdxPerStrand[s] = {}
            for chrName in posPerChr.keys():
                # Place peak on the genomic array
                try:
                    currentLen = self.chrLen[str(chrName)]
                except KeyError:
                    print(f"Warning: chromosome {str(chrName)} is not in genome annotation and will be removed")
                    continue
                array = np.zeros(currentLen, dtype="float32")
                peakIdx = posPerChr[chrName]
                np.add.at(array, peakIdx[:, 2],1)
                if not perPeakDensity:
                    # Smooth peak density
                    smoothed = oaconvolve(array, gaussian(windowSize, sigma), "same")
                    # Split consensuses
                    separators = argrelextrema(smoothed, np.less_equal)[0]      # Get local minimas
                else:
                    smoothed = np.zeros(currentLen, dtype="float32")
                    for i in range(len(peakIdx)):
                        peakSigma = (peakIdx[i, 1] - peakIdx[i, 0])*sigma
                        windowSize = int(8*peakSigma)+1
                        start = max(peakIdx[i, 2] - int(windowSize/2), 0)
                        end = min(peakIdx[i, 2] + int(windowSize/2) + 1, currentLen)
                        diffStart = max(-peakIdx[i, 2] + int(windowSize/2), 0)
                        diffEnd = windowSize + min(currentLen - peakIdx[i, 2] - int(windowSize/2) - 1, 0)
                        smoothed[start:end] += gaussian(windowSize, peakSigma)[diffStart:diffEnd]
                    separators = argrelextrema(smoothed, np.less_equal)[0]      # Get local minimas
                separators = separators[np.where(np.ediff1d(separators) != 1)[0]+1]    # Removes consecutive separators (because less-equal comparison)
                separators = np.insert(separators, [0,len(separators)], [0, currentLen])        # Add start and end points
                # Genome position separators
                sepPerStrand[s][chrName] = separators
                # Peak index separator
                array = array.astype("int16", copy=False)
                sepIdxPerStrand[s][chrName] = np.cumsum([np.sum(array[separators[i]: separators[i+1]]) for i in range(len(separators)-1)], dtype="int64")
                del array
        ########### Create matrix and consensus genomic locations ########### 
        self.matrix = []
        self.consensuses = []
        j = 0
        # Iterate over each strand
        for s in self.df.keys():
            self.df[s].sort_values(by=[0, 4], inplace=True)
            posPerChr = dict([(k, x.values) for k, x in self.df[s].groupby(0)])
            # Iterate over each chromosome
            for chrName in posPerChr.keys():
                try:
                    separators = sepPerStrand[s][chrName]
                except:
                    continue
                splits = np.split(posPerChr[chrName], sepIdxPerStrand[s][chrName])
                for i in range(len(splits)):
                    currentConsensus = splits[i]
                    # Exclude consensuses that are too small
                    if len(currentConsensus) < minOverlap:
                        continue
                    currentSep = separators[i:i+2]
                    # Setup consensuses coordinates
                    consensusStart = max(np.min(currentConsensus[:,1]), currentSep[0])
                    consensusEnd = min(np.max(currentConsensus[:,2]), currentSep[1])
                    consensusCenter = int(np.mean(currentConsensus[:,4]))
                    # Mean value of present features
                    meanScore = len(currentConsensus)
                    # Assign scores for each experiment to the current consensus
                    features = np.zeros(len(self.labels), dtype="float32")
                    features[currentConsensus[:, 6].astype(int)] = currentConsensus[:, 5].astype(float)
                    # Add consensus to the score matrix and to the genomic locations
                    self.matrix.append(features)
                    data = [chrName, consensusStart, consensusEnd, j, 
                            meanScore, s, consensusCenter, consensusCenter + 1]
                    self.consensuses.append(data)
                    j += 1
        self.matrix = np.array(self.matrix)
        if self.score == "binary":
            self.matrix = self.matrix.astype(bool)
        self.consensuses = pd.DataFrame(self.consensuses)


    def writePeaks(self):
        """
        Write matrix, datasets names and consensuses genomic locations. 
        The matrix (consensuses, datasets) uses a sparse matrix market format 
        and is saved into "matrix.mtx".
        The dataset names corresponding to the rows are saved in "datasets.txt"
        The genomic locations associated to each consensus are located in "consensuses.bed"

        Parameters
        ----------

        """
        self.consensuses.to_csv(self.outputPath + "consensuses.bed", 
                                sep="\t", header=False, index=False)
        mmwrite(self.outputPath + "matrix.mtx", csr_matrix(self.matrix.astype(float)))
        pd.DataFrame(self.labels).to_csv(self.outputPath + "datasets.txt", 
                                         sep="\t", header=False, index=False)
        with open(self.outputPath + "dataset_stats.txt", "w") as f:
            f.write("Average peak size\t" + str(self.avgPeakSize) + "\n")
            f.write("Number of Peaks\t" + str(self.numElements) + "\n")
            f.write("Number of consensus peaks\t" + str(self.matrix.shape[0]) + "\n")
            f.write("Number of experiments\t" + str(self.matrix.shape[1]) + "\n")
            f.write("Genome size\t" + str(self.genomeSize) + "\n")




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="A simple tool to integrate and analyse peak called experiments. Use the python API for more options and a more flexible usage.")
    parser.add_argument("genomeFile", help="Path to tab separated (chromosome, length) annotation file.", type=str)
    parser.add_argument("folderPath", help="Path to either folder with ONLY the peak files or comma separated list of files. Accepts compressed files.", type=str)
    parser.add_argument("fileFormat", help=
                        """
                        "bed" or "narrowPeak"
                        Format of the files being read.
                        Bed file format assumes the max signal column to be at the 6th column 
                        (0-based) in absolute coordinates.
                        The narrowPeak format assumes this to be the 9th column with this number 
                        being relative to the start position.""", type=str)
    parser.add_argument("outputFolder", help="""
                        Path of a folder (preferably empty) to write into. Will attempt to create the folder
                        if it does not exists.
                        """, type=str)
    parser.add_argument("--forceUnstranded", help=
                        """
                        If enabled, assumes all peaks are not strand-specific.
                        """, action="store_true")
    parser.add_argument("--inferCenter", action="store_true",
                        help="If enabled will use the position halfway between start and end positions as summit. Enable this only if the summit position is missing.")
    parser.add_argument("--sigma", help=
                        """
                        float or "auto_1" or "auto_2" (optional, default "auto_1")
                        Size of the gaussian filter for the peak separation 
                        (lower values = more separation).
                        Only effective if perPeakDensity is set to False. 
                        "auto_1" automatically selects the filter width based on the average
                        peak size.
                        "auto_2" automatically selects the filter width based on the peak 
                        count and genome size.
                        """, default="auto_1")
    parser.add_argument("--scoreMethod", 
                        help="""
                        "binary" or integer (default "binary")
                        If set to binary, will use a binary matrix (peak is present or absent).
                        Otherwise if set to an integer this will be the column index (0-based)
                        used as score.""", default="binary")
    parser.add_argument("--minOverlap", help=
                        """
                        integer (optional, default 2)
                        Minimum number of peaks required at a consensus.
                        2 is the default but 1 can be used on smaller datasets.
                        """, type=int, default=2)
    args = parser.parse_args()
    try:
        os.mkdir(args.outputFolder)
    except:
        pass
    with open(args.outputFolder + "command.sh", "w") as f:
        f.write(" ".join(sys.argv))
    # Run analyses
    merger = peakMerger(args.genomeFile, outputPath=args.outputFolder, 
                        scoreMethod=args.scoreMethod)
    merger.mergePeaks(args.folderPath, forceUnstranded=args.forceUnstranded, 
                      sigma=args.sigma, inferCenter=args.inferCenter, 
                      fileFormat=args.fileFormat, minOverlap=args.minOverlap)
    merger.writePeaks()
    print(f"Got a matrix of {merger.matrix.shape[0]} consensuses and {merger.matrix.shape[1]} experiments")
    sys.exit()

