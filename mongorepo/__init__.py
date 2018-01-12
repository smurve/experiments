from .posts import PostsRepo
import os
import logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger("mongorepo")


MONGO_URL = os.environ['MONGO_URL']
logger.info("connecting to mongo db at %s" % MONGO_URL)
posts = PostsRepo(MONGO_URL)
