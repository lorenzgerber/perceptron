from sys import exit

class Dataset:
    
    def __init__( self, file_name ):
        f = open( file_name, "r")
        lines = f.readlines()
        f.close()

        dimensions = str.split( lines[2] )
        self.number_of_rows = int(dimensions[1])
        self.number_of_cols = int(dimensions[2])
        self.number_of_images = int(dimensions[0])
        self.digits = [ int(i) for i in list(dimensions[3])]
        self.image_data = lines[3:]

    def loadLabels( self, file_name ):
        f = open( file_name, "r")
        lines = f.readlines()
        f.close

        dimensions = str.split( lines[2])
        if int(dimensions[0]) != self.number_of_images:
            print("number of labels does not correspond to number of images")
            exit(0)
        self.labels = lines[3:]
                    

    def printImage(self, index_of_image ):
        image = str.split(self.image_data[ index_of_image ])
        for i in range(0,self.number_of_rows):
            print( ''.join(map(str, image[((i * self.number_of_rows) + 0):((i * self.number_of_rows) + self.number_of_cols)] )))
