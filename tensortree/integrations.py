from tensortree import TensorTree


def from_nltk(tree) -> TensorTree:
    """

    :param tree: a NLTK Tree object or a string
    :param parent: a parent astlib Tree
    :return:
    """
    raise NotImplementedError("Adapt code below")
    from nltk.tree import Tree as NLTKTree

    parents = []

    def traverse(tree):
        if isinstance(tree, NLTKTree):
            root = cls(
                name=tree.label(),
                _is_terminal=False,
                parent=parent
            )
            for node in tree:
                cls.from_nltk(node, parent=root)

        else:
            cls(
                name=tree,
                _is_terminal=True,
                parent=parent
            )
    return root



from spacy.tokens.doc import Doc


def from_spacy(sentence: Doc) -> TensorTree:
    """

    :param tree: a NLTK Tree object or a string
    :param parent: a parent astlib Tree
    :return:
    """
    parents = []
    node_data = []

    def traverse(token, parent=-1):
        print(token.text)
        data = {
            "token": token.text,
            "position": token.head.pos_,
            "relation": token.dep_,

        }
        node_data.append(data)
        parents.append(parent)

        node_idx = len(parents) - 1

        for child in token.children:
            traverse(child, parent=node_idx)

    traverse(sentence.root)
    print(parents)

    return TensorTree.from_array(node_data=node_data, parents=parents)