import nltk
class preprocess:
    def remove_stopwords(self,textstring):
        """Preprocesses a string by removing stop words, symbols, digits, etc.
        outputs a list of tokens."""
        
        # assert that a string was passed
        assert isinstance(textstring,str) , "THIS IS NOT A STRING"
        # parse string into array of words
        words = textstring.split(" ")
        # remove stopwords
        words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
        return words


    def preprocess_dataset(self):
        """Preprocess dataset by removing stopwords"""
        # create output files
        test_data_output_file = open("dataset/test_formatted.csv", mode="x")
        train_data_output_file = open("dataset/train_formatted.csv", mode="x")
        # read raw data
        test_data_raw = open("dataset/test_amazon.csv")
        train_data_raw = open("dataset/train_amazon.csv")

        for line in test_data_raw:
            addnewline =True
            line = line.split(",")
            rmstop = self.remove_stopwords(line[1])
            test_data_output_file.write(f"{line[0]}, ")
            for x in rmstop:
                x.replace(",","")
                if x.count("\n")>0:
                    addnewline = False
                test_data_output_file.write(f"{x.lower()} ")
            if(addnewline):
                test_data_output_file.write("\n")
        for line in train_data_raw:
            addnewline=True
            line = line.split(",")
            rmstop = self.remove_stopwords(line[1])
            train_data_output_file.write(f"{line[0]}, ")
            for x in rmstop:
                x.replace(",","")
                if x.count("\n")>0:
                    addnewline = False
                train_data_output_file.write(f"{x.lower()} ")
            if(addnewline):
                train_data_output_file.write("\n")
            



