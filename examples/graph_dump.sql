CREATE TABLE edges (
  source TEXT,
  target TEXT,
  relation TEXT
);

INSERT INTO edges (source, target, relation) VALUES
  ('Knowledge Graph', 'Claim', 'supports'),
  ('Retrieval-Augmented Generation', 'Citation', 'improves'),
  ('Multi-agent System', 'Claim', 'organizes');

CREATE TABLE figures (
  id TEXT,
  caption TEXT,
  path TEXT,
  related_entities TEXT
);

INSERT INTO figures (id, caption, path, related_entities) VALUES
  ('fig-kg-overview', 'Knowledge graph schema overview', 'figures/kg-overview.png', 'Knowledge Graph,Claim');
