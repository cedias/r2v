# coding: utf8
import sqlite3


class Database(object):
    """
    Holds all the data for recommendation:
    Contains two tables:
    Table Review(user,item,review,rating,timestamp,test)
    Table Sentences(id_sent,user,item,rating)
    """

    def __init__(self, name):
        self.name = name
        self.con = sqlite3.connect(self.name)

    @staticmethod
    def build(db_name, data_generator):
        con = sqlite3.connect("{}".format(db_name))
        c = con.cursor()
        # Create table
        c.execute('''CREATE TABLE reviews
                 (item integer, user integer, review text, rating real, r_timestamp timestamp ,test boolean)''')
        c.execute('''CREATE TABLE sentences
                 (id_sent integer, item integer, user integer, sent text, rating real)''')
        # Save (commit) the changes
        con.commit()
        c.executemany('INSERT INTO reviews VALUES (?,?,?,?,?,?)', data_generator)
        con.commit()

        db = Database(db_name)
        c.executemany('INSERT INTO sentences VALUES (?,?,?,?,?)', db._generate_sentences())
        con.commit()

        db._create_indexs()

        return db

    ## Private
    def _generate_sentences(self):
        i=0
        for item,user,review,rating,_ in self.getFullReviews():
            for sent in review.split("."):
                if len(sent) > 2:
                    yield (i, item,user,sent.strip(),rating)
                    i+=1


    def _create_indexs(self):
        c = self.con.cursor()
        c.execute("CREATE index IF NOT EXISTS uind ON reviews(user)")
        c.execute("CREATE index IF NOT EXISTS iind ON reviews(item)")
        c.execute("CREATE index IF NOT EXISTS rind ON reviews(rating)")
        c.execute("CREATE index IF NOT EXISTS teid ON reviews(test)")
        c.execute("CREATE index IF NOT EXISTS tid ON reviews(r_timestamp)")

        c.execute("CREATE index IF NOT EXISTS sind ON sentences(id_sent)")
        c.execute("CREATE index IF NOT EXISTS uind ON sentences(user)")
        c.execute("CREATE index IF NOT EXISTS iind ON sentences(item)")
        c.execute("CREATE index IF NOT EXISTS rind ON sentences(rating)")
        self.con.commit()

    ## Overall Methods

    def getOverallBias(self):
        c = self.con.cursor()
        c.execute("SELECT avg(rating) as bias FROM reviews WHERE not test")
        return c.fetchone()

    ## User Methods

    def getAllUsers(self):
        c = self.con.cursor()
        c.execute("SELECT user,count(*) as cp, avg(rating) as biais FROM reviews GROUP BY user ORDER BY cp DESC ")
        return c.fetchall()

    def getUserBias(self, user):
        c = self.con.cursor()
        c.execute("SELECT avg(rating) as bias FROM reviews WHERE user = {} and not test group by user".format(user))
        return c.fetchone()[0]

    def getUsersBias(self):
        c = self.con.cursor()
        c.execute("SELECT user,avg(rating) as bias FROM reviews WHERE not test group by user")
        return c.fetchall()

    def getUserReviews(self, user, test):
        c = self.con.cursor()
        if test is True:
            c.execute("SELECT item,rating,review FROM reviews WHERE user = {} and test".format(user))
        elif test is False:
            c.execute("SELECT item,rating,review FROM reviews WHERE user = {} and not test".format(user))
        else:
            raise AttributeError("Argument test is either True or False: here test is {}".format(test))
        return c.fetchall()

    ## Item Methods

    def getAllItems(self):
        c = self.con.cursor()
        c.execute("SELECT item,count(*) as cp, avg(rating) as biais FROM reviews GROUP BY item ORDER BY cp DESC ")
        return c.fetchall()

    def getItemBias(self, item):
        c = self.con.cursor()
        c.execute("SELECT avg(rating) as bias FROM reviews WHERE item = {} and not test group by item".format(item))
        return c.fetchone()[0]

    def getItemsBias(self):
        c = self.con.cursor()
        c.execute("SELECT item, avg(rating) as bias FROM reviews WHERE not test group by item")
        return c.fetchall()

    def getItemReviews(self, item, test):
        c = self.con.cursor()
        if test is True:
            c.execute("SELECT user,rating,review FROM reviews WHERE item = {} and test".format(item))
        elif test is False:
            c.execute("SELECT user,rating,review FROM reviews WHERE item = {} and not test".format(item))
        else:
            raise AttributeError("Argument test is either True or False: here test is {}".format(test))
        return c.fetchall()

    # Review Methods

    def getReviewRating(self, item, user):
        c = self.con.cursor()
        c.execute("SELECT user,item,rating FROM reviews WHERE item = {} and user = {}".format(item, user))
        return c.fetchone()

    def getAllReviews(self, test):
        c = self.con.cursor()
        if test is True:
            c.execute("SELECT item,user,rating FROM reviews WHERE test")
        elif test is False:
            c.execute("SELECT item,user,rating FROM reviews WHERE not test")
        elif test == None:
            c.execute("SELECT item,user,rating FROM reviews")
        else:
            raise AttributeError("Argument test is either True or False or None: here test is {}".format(test))

        return c.fetchall()

    def getFullReviews(self):
        c = self.con.cursor()
        c.execute("SELECT item,user,review,rating,r_timestamp,test FROM reviews")
        return c.fetchall()
