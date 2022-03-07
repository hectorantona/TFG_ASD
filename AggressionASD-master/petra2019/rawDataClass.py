


class RawData:
    subjectIDs = {}
    featuresForAllUsers = {}

    def __init__(self):

    def readAllSubjectIDsFromRootDirectory(self, rootDir, pathStyle='/', idNDigits=4):
        self.subjectIDs = getSubjectIdsDictionaryFromDirectoryList(listFoldersInDir(rootDir), pathStyle, idNDigits)


    @property
    def list_folders_in_dir(dir_path):
        # list all folders including the current path
        folders = [x[0] for x in os.walk(dir_path)]
        # delete current path from list
        # del folders[0]
        return folders[1:]

    # uIDDict = getUserIdsDictionaryFromDirectoryList(dirList, pathStyle, idNDigits):
    # where uIDDict is a dictionary {newID,UID} where UID is a list with the
    # different unique user codes, and newID is a contiguous positive integer number
    # (1 to NumberOfUsers) associated with each UID.
    # Inputs: dirList a list containing all directory names;
    # pathStyle: that should be '/' for linux style or '\' for windows.
    # idNDigits: this code assumes that each user ID is identified by the
    # first idNDigits of the folders containing the csv data files.
    def getSubjectIdsDictionaryFromDirectoryList(dirList, pathStyle='/', idNDigits=4):
        # ids = [x.replace(path + pathStyle, '') for x in folders]
        SID = [x.split(pathStyle)[-1][0:idNDigits] for x in dirList]
        newID = np.unique(UID, return_inverse=True)[1];
        return dict(zip(newID + 1, UID))
