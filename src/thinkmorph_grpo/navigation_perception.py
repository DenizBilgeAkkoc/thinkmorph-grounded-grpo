"""Research-only AFTER image cell perception; no scoring or intended route."""
PROMPT = 'Inspect only the supplied image. Transcribe physically drawn route annotations, independently of whether any route is useful, legal, safe, or reaches a goal. Do not evaluate correctness. Do not invent a route from character/goal positions. Ignore red clothing and colored objects; identify added route strokes and arrowheads.\nRead grid dimensions from the image. Use zero-based [row,column]: top-left is [0,0], rows increase downward and columns increase rightward. Start at the visible drawn route tail and follow arrow direction. Count EVERY crossed CELL BOUNDARY. A long horizontal or vertical stroke can span several cells; each one-cell move needs its own adjacent visited cell, regardless of how many arrowheads or line segments are drawn. Include intermediate cells. Do not collapse a two-cell horizontal span into one move. Describe each visible boundary crossing before listing cells. If route direction or cell registration is unclear, use assessable=false; never repair an ambiguous drawing from an assumed intended solution.\nReturn exactly JSON with fields evidence, grid_rows, grid_cols, path_present, boundary_crossings, visited_cells, assessable, issues. Each boundary_crossings entry has exactly from_cell, to_cell, evidence; each cell is a two-integer [row,column] list. visited_cells is the ordered list of all adjacent cells including the tail cell. Dimensions are positive integers; path_present and assessable are booleans; issues is a string list; evidence fields are strings.\nUse path_present=false with both lists empty if no drawn route exists. Every consecutive pair must be one horizontally or vertically adjacent cell. If a drawn path visibly exits the grid, an outside adjacent cell may be included, but never infer an unseen continuation. Do not output moves, safety, hole traversal, goal success, or any corrected path. Keep evidence concise.'

def cell(value):
    return isinstance(value,list) and len(value)==2 and all(type(x) is int for x in value)

def validate_perception(value):
    if set(value)!={'evidence','grid_rows','grid_cols','path_present','boundary_crossings','visited_cells','assessable','issues'}:
        raise ValueError('Invalid perception fields')
    if type(value['assessable']) is not bool or not value['assessable']:
        raise ValueError('Unassessable route perception')
    if type(value['path_present']) is not bool or not isinstance(value['evidence'],str):
        raise ValueError('Invalid presence/evidence')
    if any(type(value[key]) is not int or value[key]<1 for key in ('grid_rows','grid_cols')):
        raise ValueError('Invalid grid dimensions')
    cells=value['visited_cells'];crossings=value['boundary_crossings']
    if not isinstance(cells,list) or any(not cell(c) for c in cells) or not isinstance(crossings,list):
        raise ValueError('Invalid route cells')
    if not isinstance(value['issues'],list) or any(not isinstance(x,str) for x in value['issues']):
        raise ValueError('Invalid issues')
    if not value['path_present']:
        if cells or crossings:raise ValueError('Absent path cannot have crossings')
        return []
    if len(cells)<2 or len(crossings)!=len(cells)-1:
        raise ValueError('Every drawn cell transition needs an explicit crossing')
    moves=[]
    for start,end,crossing in zip(cells,cells[1:],crossings):
        if set(crossing)!={'from_cell','to_cell','evidence'} or crossing['from_cell']!=start or crossing['to_cell']!=end or not isinstance(crossing['evidence'],str):
            raise ValueError('Crossings and visited cells disagree')
        delta=(end[0]-start[0],end[1]-start[1])
        if delta not in {(-1,0),(1,0),(0,-1),(0,1)}:raise ValueError('Non-adjacent cells; never interpolate a missing move')
        moves.append({(-1,0):'U',(1,0):'D',(0,-1):'L',(0,1):'R'}[delta])
    return moves
