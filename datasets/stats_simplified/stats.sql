-- Users
CREATE TABLE users (
	Id SERIAL PRIMARY KEY,
	Reputation INTEGER ,
	CreationDate BIGINT ,
	Views INTEGER ,
	UpVotes INTEGER ,
	DownVotes INTEGER
);

-- Posts
CREATE TABLE posts (
	Id SERIAL PRIMARY KEY,
	PostTypeId SMALLINT ,
	CreationDate BIGINT ,
	Score INTEGER ,
	ViewCount INTEGER,
	OwnerUserId INTEGER,
	AnswerCount INTEGER ,
	CommentCount INTEGER ,
	FavoriteCount INTEGER,
	LastEditorUserId INTEGER
);

-- PostLinks
CREATE TABLE postLinks (
	Id SERIAL PRIMARY KEY,
	CreationDate BIGINT ,
	PostId INTEGER ,
	RelatedPostId INTEGER ,
	LinkTypeId SMALLINT
);

-- PostHistory
CREATE TABLE postHistory (
	Id SERIAL PRIMARY KEY,
	PostHistoryTypeId SMALLINT ,
	PostId INTEGER ,
	CreationDate BIGINT ,
	UserId INTEGER
);

-- Comments
CREATE TABLE comments (
	Id SERIAL PRIMARY KEY,
	PostId INTEGER ,
	Score SMALLINT ,
	CreationDate BIGINT ,
	UserId INTEGER
);

-- Votes
CREATE TABLE votes (
	Id SERIAL PRIMARY KEY,
	PostId INTEGER,
	VoteTypeId SMALLINT ,
	CreationDate BIGINT ,
	UserId INTEGER,
	BountyAmount SMALLINT
);

-- Badges
CREATE TABLE badges (
	Id SERIAL PRIMARY KEY,
	UserId INTEGER ,
	Date BIGINT
);

-- Tags
CREATE TABLE tags (
	Id SERIAL PRIMARY KEY,
	Count INTEGER ,
	ExcerptPostId INTEGER
);
