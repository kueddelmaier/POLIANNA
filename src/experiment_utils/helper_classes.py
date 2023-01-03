import collections

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

class repository: 
    """
    A repository is a class that contains all the information about a specific document.
    It is used to identify a document in the dataframe and to identify the document in the file system.
    A repository is characterized by the policy, title, chapter, section, and article. 
    If the repository is a sentence repository, it also contains the sentence number.

    """
    def __init__(self, policy = None, title = None, chapter = None, section = None, article = None, sentence = None, index_name = None):
        self.policy = policy
        self.title = title
        self.chapter = chapter
        self.section = section
        self.article = article
        self.sentence = sentence
        self.index_name = index_name # the name of the repository used by the dataframe as index

    def __repr__(self): #how to print the repository to the console
        return f"policy:{self.policy} title:{self.title} chapter:{self.chapter} section:{self.section} article:{self.article}" + (f"sentence:{self.sentence}" if self.sentence != None else '')
   
    def __eq__(self, other):
        if isinstance(other, repository):
            return self.policy == other.policy and self.title == other.title and self.chapter == other.chapter and self.section == other.section and self.article == other.article and self.sentence == self.sentence
        return False
        
    def __hash__(self):
        return hash(self.__repr__())


    @classmethod
    def from_repository_name(cls, rep_str):
        """
        2nd initializer that creates a repository object directly from a repository string e.g 'EU_32008R1099_Title_0_Chapter_0_Section_0_Article_03.txt
        """                
        folder_parts = rep_str.split('_')                  #split the stchecks if the search-criteria defined in repository 'other' is matching the the current repository ring at '_' into parts 
        policy = folder_parts[0] + '_' + folder_parts[1]   #we only want to split at every 2nd '_', so merge the 1. and 2., 3. and 4. again 
        
        if folder_parts[2] in  ['front', 'Whereas']:       #exeption for the 'whereas' and 'front'
            title = folder_parts[2]
            chapter = None
            section = None
            article = None
            sentence = None
        else:
            title = folder_parts[2] + '_' + folder_parts[3]
            chapter = folder_parts[4] + '_' + folder_parts[5]
            section = folder_parts[6] + '_' + folder_parts[7]
            article = folder_parts[8] + '_' + folder_parts[9]
            

            if len(folder_parts) == 12:
                sentence = folder_parts[10] + '_' + folder_parts[11]
            else:
                sentence = None
        
        return cls(policy,title, chapter, section, article, sentence, rep_str)  #return a repository with the previously defined attributes
    
    def match(self, other):       
        """
        Checks if the attributes of the search-criteria are a subset of the attributes of the current directory (=match)
        """                                                    
        self_value_set = set([x for x in list(self.__dict__.values()) if x != None]) #creates a set of all the attributes ignoring 'None'    
        other_value_set = set([x for x in list(other.__dict__.values()) if x != None])
        
        return set(other_value_set).issubset(self_value_set) #returns True if the attributes of the search-criteria is a subset of the attributes of the current directory (=match)
    
class token:
    """ 
    A token is a single word or punctuation mark in a sentence. It has a start and stop position in the sentence and a text. 
    Furthermore it has a list of spans that labeled this token and a repository that defines the Article it belongs to.
    
    """
    def __init__(self, start, stop, text, rep):
        self.start = start
        self.stop = stop
        self.text = text
        self.rep = rep
        self.spans = []     #spans that labeled this token

    def add_span(self, span_id):
        self.spans.append(span_id)

    
    def get_token_spans(self, annotators = 'Curation'):
        """ Returns a list of spans that labeled this token """

        if annotators == 'all':
            return self.spans

        if annotators == 'annotators':
            return [span_ for span_  in self.spans if 'Curation' not in span_.annotator]

        if annotators == 'Curation': 
            return [span_ for span_  in self.spans if 'Curation' in span_.annotator]
            
        else:
            return [span_ for span_  in self.spans if annotators in span_.annotator]


    def get_token_tags(self, annotators = 'Curation'):
        """ Returns a list of tags for the token """
        return [span_.tag for span_ in self.get_token_spans(annotators)]

    def get_span_count(self, annotators = 'Curation'):
        """ Returns the number of spans that labeled this token """
        return len(self.get_token_spans(annotators))
        
    def __repr__(self):
        return f"start:{self.start} stop:{self.stop} text:{self.text} tag_count:{self.get_span_count(annotators ='Curation')}"

    def __hash__(self): # used to remove duplicates
        return hash((self.start, self.stop, self.text))
        
class span:
    """ 
    A span is a labeled part of a sentence. It has a start and stop position in the sentence and a text.
    Furthermore it has a list of tokens that are part of this span and a repository that defines the Article it belongs to.
    The label of each span is characterized by a layer, a feature and a tag.

    """
    def __init__(self, span_id = None, layer = None, feature = None, tag = None, start = None, stop = None, text = None, tokens = None, rep = None, annotator = None):
        self.span_id = span_id
        self.layer = layer
        self.feature = feature
        self.tag = tag
        self.start = start
        self.stop = stop
        self.text = text
        self.tokens = tokens
        self.rep = rep
        self.annotator = annotator


    def __eq__(self, other): 
        if isinstance(other, span): # don't attempt to compare against unrelated types
            return self.span_id == other.span_id
        return False

    
    def __repr__(self): #for debugging purpose, defines how object is printed
        return f"span id:{self.span_id} annotator:{self.annotator} layer:{self.layer} feature:{self.feature} tag:{self.tag} start:{self.start} stop:{self.stop} text:{self.text}"

    def __hash__(self):
        return hash((self.annotator, self.layer, self.feature, self.tag, self.start, self.stop, self.text, self.rep.__hash__())) #tags with identical properties and identical repo shoudl yield the same hash so they can be removed

    def exact_match(self, other):
        return self.start == other.start and self.stop == other.stop and self.tag_ == other.tag_

    def partial_match(self, other):
        return self.start < other.stop and other.start < self.stop and self.tag_ == other.tag_
    
    def tokenwise_f1_score(self, other):
        """
        Calculates the f1 score between two spans based on the shared tokens.

        """

        if self.tag_ != other.tag_:
            return 0
        if self.tag_ == None or other.tag_ == None:
            return 0
        if self.tag_ == other.tag_:
            if len(self.tokens) == 0 or len(other.tokens) == 0:
                #if either is no answer, return 1 if they agree, zero else
                return int(self.tokens == other.tokens)

            common = collections.Counter(self.tokens) & collections.Counter(other.tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0

            precision = 1.0 * num_same / len(self.tokens)
            recall = 1.0 * num_same / len(other.tokens)
            f1 = (2*precision * recall) / (precision + recall)
            return f1
        else:
            raise ValueError('None of the mentioned')




