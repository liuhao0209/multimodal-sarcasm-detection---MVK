from transformers import BertTokenizer, pipeline
sentence_sentiment_polarity = pipeline("sentiment-analysis")
#按顺序放入词就可以了  5379
from tqdm import tqdm


label = []
text_list = ['Like','sculpting','Marge', 'Chandy', 'chose', 'required', 'while', 'exploded', 'squeaky', 'followed', 'peeking', 'coffees', 'nicer', 'things', 'deadened', \
             'suggestions', 'groundrules', 'another', 'At', 'under', 'ourselves', 'relied', 'Hurley', 'would', 'peeing', 'cellular', 'upstate', 'Bye_bye', 'clingy', 'wrote', \
             'usin', 'lullabye', 'Tooty', 'your', 'cou', 'No', 'Snuggles', 'girlfriends', 'chicks', 'Tupolo', 'Jeannie', 'chandler', 'goop', 'Wants', 'drowning', 'realised', 'Swiss', 'Popes', 'sends', 'balanced', 'updating', 'birthfather', 'paramedics', 'occurring', 'apron', 'thetoughest', 'macaws', 'thing', \
             'Blepe', 'glazing', 'Like', 'PAJAMAS', 'thinkin', 'Where', 'enjoying', 'Partners', 'Yeah_well_yeah', 'colon', 'nearly', 'gelatenous', 'listing', 'good_night', 'Rappelano', 'Drew', 'auditioning', 'Victoria',  \
             'Celebrities', 'longshot', 'Clogs', 'where', 'asked', 'Saltines', 'squat', 'riding', 'understudy', 'condoms', 'yumm', 'almost', 'spinning', 'darnit', 'Somerfield', 'Brides', 'Carols', 'over_pronouncing', 'Norman', \
             'um_hmm', 'Winky', 'either', 'name', 'we', 'uncalled', 'Sweety', 'sort', 'qualifies', 'Med', 'happened', 'Ohhhhh', 'sav', 'wenus', 'allways', 'crumbles', 'got', 'pate', 'signing', 'F_hah', 'yet', 'Tonight', \
             'datable', 'tinted', 'Ooooohh', 'object', 'doof', 'No_no', 'until', 'shaken', 'cervix', 'asks', 'meant', 'Dean', 'Interiors', 'regulations', 'pearl', \
             'otherwise', 'Into', 'Almost', 'Marty', 'Capades', 'Shes', 'into', 'Westburg', 'that', 'seen', 'moisturizer', 'Narnia', 'picked', 'besides', 'Sidney', 'indicated', 'ex_wife', 'ask', 'Holden', 'Sergio', 'days', 'their', \
             'didnt', 'Gone', 'Sapien', 'Hormones', 'Minnie', 'Jason', 'siadic', 'Good_night', 'Scuse', 'Ross', 'also', 'discussed', 'Lifeguards', 'Poconos', 'casa', 'thingys', 'leeeeean', 'We', 'we-we', 'Boutros', 'd\'know', 'hired', \
             'roll_away', 'Wheres', 'Annabelle', 'graduated', 'Karen', 'Unless', 'headin', 'Although', 'hypnotised', 'implementing', 'Forty_five', 'Allesandro', 'gathered', 'Algonquin', 'Timothy', 'who', 'how', 'canned', 'Filing', 'used', 'he', 'its', \
             'good_good', 'Me', 'Cheers', 'factor', 'heard', 'FDA', 'sure_sure', 'And', 'said-I', 'classics', 'Lecter', 'turned', 'Anytime', 'and', 'reattaching', 'Peter', 'supergay', 'bubbling', 'watching', 'eavesdropping', 'Chicago', 'This', \
             'sweepin', 'Who', 'suspected', 'nothing', 'transferring', 'huh-uh', 'Oops', 'Bonnie', 'brackety', 'Or', 'nothing', 'Rostins', 'rehearsals', 'sooo', 'Innsbruck', 'Wh_what', 'Now', 'dingus', 'lit', 'spied', 'tourists', 'Basinger', \
             'panchetta', 'checked', 'These', 'implantation', 'if', 'pouty', 'legs', 'Casey', 'half_hour', 'sightless', 'Congratu', 'Cushions', 'complaints', 'kinds', 'Turned', 'realizing', 'Magioni', 'kissing', 'whoever', 'realilized', 'tracked', 'Barrymore', 'realises', \
             'coyotes', 'That', 'Hi', 'Lips', 'Says', 'January', 'Waitressing', 'say', 'sold', 'Um_hmm', 'tweezers', 'Cinderelly', 'is', 'which_which', 'basset', 'drying', 'stayed', 'feet', 'banging', 'indian', 'no', 'York', 'Laude', 'these', \
             'tree', 'dumped', 'Pheebs', 'returning', 'Geller', 'ah', 'wait', 'judged', 'licked', 'wrapped', 'corpses', 'Women', 'beind', 'basing', 'earlier', \
             'Minsk', 'Lives', 'regimented', 'strippers', 'slaughtered', 'yourself', 'Sergei', 'wait', 'retiling', 'You', 'Abbey', 'themselves', 'know', 'Pepperidge', 'systems', \
             'periscope', 'Franzblau', 'Tonight', 'ankles', 'Bactine', 'shunned', 'stairs', 'evening', 'He', 'by', 'Tribbiani', 'Leia', 'self_cleaning', 'NO', 'neighbourhood', \
             'know', 'bein', 'By', 'Mailer', 'Sometimes', 'Greenstein', 'tetanus', 'life-life', 'name', 'kinda', 'substances', 'during', 'lutely', 'why', 'Aaaaah', \
             'than', 'cabinets', 'depriving', 'Michener', 'held', 'pipe-fitting', 'cuter', 'Anybody', 'betcha', 'Embryossss', 'G_good', 'too', 'shoulda', 'skiing', 'stand_up', \
             'hitting', 'suspected_ah', 'Yeeees', 'Trapp', 'Greek', 'Heard', 'they', 'lived', 'turnin', 'Ah_ha_ha', 'enemas', 'originated', 'peed', 'Paul', 'she', \
             'nevermind', 'losers', 'hmmm', 'across', 'outloud', 'yep', 'reps', 'mignon', 'himself', 'shaving', 'like_like', 'dancy', 'Treeger', 'Ohhhhhhh', 'lips', \
             'co_worker', 'Pennsylvania', 'quantities', 'Baddest', 'Vic', 'admissions', 'switched', 'smiling', 'salads', 'Damnit', 'wait', 'Fuggetaboutit', 'debloons', 'Gram', 'receptions', 'Parents', \
             'what', 'name', 'till', 'belive', 'Casa', 'field', 'bullets', 'owes', 'Makes', 'deconstructing', 'barium', 'Roger', 'this', 'Doubtfire', 'starin', \
             'ageist', 'showed', 'capades', 'now', 'centimeters', 'drivin', 'Russian', 'anywhere', 'explained', 'wait', 'Douglas', 'nooo', 'thy', 'Trident', 'someone', \
             'eats', 'pre_appetizer', 'Sean', 'walkin', 'said', 'counted', 'scrud', 'Denise', 'Goodacre', 'both', 'Which', 'Be', 'players', 'Got', 'hovercrafts', \
             'hooking', 'since', 'appalling', 'France', 'expecting', 'Pitstains', 'ME', 'Petrie', 'About', 'something-or', 'like', 'Donna', 'given', 'hi', 'Drake', \
             'him', 'laminated', 'minutes', 'England', 'fattest', 'translated', 'Hurely', 'barley', 'finding', 'them', 'speacial', 'The_the', 'that_that', 'with', 'ahhh', \
             'will', 'Matches', 'spectre', 'sailed', 'unless', 'can', 'Salem', 'Now', 'righteous', 'thrown', 'filing', 'smokey', 'Spiderman', 'Mon', 'coincidences', \
             'lovers', 'Anyway', 'our', 'spelled', 'hon', 'if_if', 'Phoebs', 'Broadway', 'sliced', 'Is', 'Floyd', 'behaviour', 'Wherere', 'blacked', 'also_also', \
             'starting', 'grew', 'dented', 'more', 'hormones', 'chickened', 'Sapiens', 'grown', 'Th_this', 'mattresses', 'Mike', 'Chuckles', 'goggles', 'Theres', 'protects', 'here', \
             'results-Whoa', 'fonts', 'ho', 'my', 'am', 'addressed', 'she', 'though', 'countdown', 'helped', 'breakin', 'some', 'Guys', 'one_Dude', 'clients', 'anybody', 'officiating', \
             'Wiper', 'haaaah', 'remembered', 'Later', 'remaking', 'knocked', 'lads', 'Gogh', 'kicked', 'Whats', 'Y\'miss', 'so', 'sicker', 'checking', 'caved', 'volcanoes', \
             'practising', 'critics', 'Yep', 'was', 'straigh', 'Ingrid', 'Chandler', 'scariest', 'picturing', 'pademarie', 'someday', 'Yours', 'suckin', 'Boston', 'Korean', \
             'Cookies', 'sent', 'classifieds', 'pizzas', 'spewing', 'mixing', 'An', 'Heston', 'after', 'heh', 'how_how', 'shutting', 'whoo_hoo', 'you', 'Becasue', \
             'somewhere', 'smeared', 'Watcha', 'San', 'Looks', 'cover_up', 'losin', 'drove', 'India', 'Ever', 'Lauren', 'Velula', 'craziest', 'Thats', 'Matthews', 'diffency', \
             'yankin', 'might', 'coaching', 'Hey_hey', 'Another', 'Sliced', 'insisted', 'ups', 'off_stage', 'dorks', 'More', 'Oberman', 'sweetums', 'responsibilities', 'riboflavin', \
             'adults', 'shoop', 'Somewhere', 'andoh', 'Rhonda', 'Mindy', 'Mira', 'Kinda', 'Mannequins', 'whether', 'that', 'Julie', 'talkin', 'You_you', 'Mendels', \
             'slid', 'humping', 'belongs', 'erotiery', 'clogging', 'and', 'massager', 'unclogging', 'chestnuts', 'DeMarco', 'wonerful', 'seagulls', 'Isabella', 'Jesus', 'showing', 'Troopers', \
             'toughest', 'debating', 'Adios', 'Michael', 'turning', 'Times', 'whose', 'doody', 'spoiled', 'tellin', 'Geller_Willick_Bunch', 'name', 'too', 'spacecrafts', 'organisms', \
             'appreciates', 'street', 'Rollerson', 'Paris', 'Chrysler', 'That', 'Remoray', 'spun', 'packing', 'Bing', 'how', 'Balls', 'pressed', 'tomorow', 'B-bye', \
             'honing', 'Hello', 'makin', 'upstairs', 'wonderfulness', 'Done', 'never', 'fell', 'Bazida', 'Charla', 'heartbeats', 'Trib', 'stard', 'Freddie', 'Marks', \
             'gazillion', 'Orient', 'Waxine', 'AAAAHHHHHH', 'Winona', 'Shielding', 'Dorothy', 'HeyHey', 'Barley', 'scientists', 'Since', 'threatened', 'nodded', 'vulcanised', 'be', \
             'looked', 'name', 'lobsters', 'Heckles', 'They', 'Carney', 'Rushmore', 'chipped', 'Collins', 'cutie', 'See_see', 'Pffffffft', 'Stephanie', 'poking', 'whatcha', 'did', \
             'terriffic', 'Tso', 'Finders', 'bill_maybe', 'gettin', 'Floopy', 'Mine', 'Most', 'boarding', 'hamsters', 'designers', 'Exellent', 'Chunkys', 'Spinning', 'acutally', 'bionic', \
             'whats', 'Did', 'fishtachios', 'haveth', 'viking', 'So', 'committees', 'term', 'Grievances', 'wore', 'Sarah', 'On', 'guest', 'dentists', 'ordered', \
             'countries', 'Pittain', 'filled', 'movin', 'Tuesday', 'WE', 'added', 'taunting', 'Why', 'mealworms', 'howd', 'woulda', 'What', 'me', \
             'if', 'tonight', 'folders', 'snoring', 'whyyy', 'sometimes', 'backed', 'trading', 'pulling', 'What\'dya', 'break_in', 'Where', 'Eyes', 'Sounds', 'Hammel', 'guys_guys_guys', \
             'shots', 'Americccan', 'pilgrim', 'eyelashes', 'MacDonald', 'laughing', 'When', 'Eggert', 'regulars', 'burgundy', 'such', 'may', 'waving', 'seventies', 'Their', 'Nodded', \
             'impressivo', 'fling', 'processors', 'had', 'scenarios', 'spent', 'May', 'liquorice', 'but', 'met', 'your_your', 'heading', 'Messier', 'not', 'Tegrin', \
             'October', 'drew', 'Carin', 'imagining', 'safer', 'perked', 'Does', 'tended', 'taught', 'yogart', 'name', 'mentioning', 'jillion', 'Hola', 'Amish', \
             'rescedule', 'muffins', 'jewellers', 'midgets', 'languages', 'Your', 'calling', 'Chatracus', 'Acres', 'Bermuda', 'gram', 'pointing', 'Japanese', 'later', \
             'came', 'Nono', 'twice', 'Rastatter', 'Middletown', 'built', 'Clydesdales', 'Richard', 'Our', 'Francisco', 'Could', 'felt', 'taken', 'Spock', 'neurosurgeon', \
             'flung', 'why', 'wait', 'filet', 'Knicks', 'You', 'G.I', 'Staton', 'welling', 'Wednesday', 'hah_hah', 'self_destructive', 'perhaps', 'yourselves', 'recarpet', 'Sometimes', \
             'spilled', 'Sallidor', 'penis', 'pistachios', 'Gunnersens', 'Days', 'weeks', 'forty_five', 'Baldwin', 'found', 'German', 'watch', 'Ahh', 'Sometime', 'Its', \
             'She', 'bing_bong', 'truth_You', 'Aaaah', 'albums', 'There', 'Ahhh', 'After', 'minites', 'Gettin', 'positive', 'blouses', 'loaner', 'Windkeeper', 'us', \
             'Bea', 'compartments', 'impressions', 'How', 'Vesuvius', 'gim', 'Grazie', 'supposed', 'what\'s_what', 'Shall', 'Chico', 'aprons', 'Something', 'Whoa', 'What', \
             'Who', 'bing', 'good_ bye', 'straddling', 'Bergman', 'Brittany', 'Canadian', 'Stacy', 'Getting', 'identifies', 'yanking', 'andndash', 'If', 'started', 'have', \
             'fulfilling', 'you', 'if', 'name', 'Africa', 'doesnt', 'Emily', 'Terry', 'make', 'called', 'Losing', 'Bryce', 'Hon', 'members', 'even', \
             'rather', 'Ooh_hoo', 'Swedish', 'Brooklyn', 'Pudo', 'Such', 'ugliest', 'Dudley', 'sounded', 'Livingston', 'wait', 'Even', 'itself', 'Had', 'redoing', 'hoping',
             'cornered', 'Aaaaahhh', 'mockolate', 'Goldman', 'sneakin', 'sido', 'Thursday', 'impressed', 'once', 'gon', 'passed', 'Someone', 'Anything', 'judging', 'before', 'Never_never', 'favour', 'Ma', 'ours', 'accross', \
             'laughed', 'gravy', 'fest', 'Nothing', 'should', 'hung', 'Nobel', 'ahhhhhh', 'grrreat', 'Feburary', 'Esther', 'Olympics', 'letting', 'timer', 'prep', 'As', 'schmush', 'Condoms', 'can', \
             'It', 'rightie', 'as', 'Can', 'the_the', 'theme', 'Elizabeth', 'jullienne', 'emergencies', 'rommmates', 'embryos', 'softener', 'Was', 'Thy', 'always', 'Seasame', 'Indians', 'beepers', 'Marshall', \
             'Shannon', 'Helen', 'Lambs', 'Italian', 'here_', 'creepin', 'seems', 'oneself', 'deserves', 'o\'clock', 'him', 'Upstate', 'nouse', 'gone', 'listening', 'Starbucks', 'pressing', 'hygienist', 'Skates', \
             'from', 'indeedy', 'lookin', 'No', 'est', 'Touchet', 'CREDITS', 'From', 'Aaaaagggghhhhh', 'fellas', 'yours', 'Doing', 'Whaddyou', 'attended', 'Weddings', 'went', 'Cities', 'come_here', 'Op', \
             'penache', 'bringing', 'copying', 'eventhough', 'midterms', 'Would', 'lochs', 'Beth', 'bitemebitemebitemebiteme', 'lamps', 'must', 'not_not', 'boobies', 'Neither', 'Clockwork', 'nudes', 'germs', 'painted', 'applying', '', \
             'filming', 'Barbara', 'left', 'Steffi', 'lovin', 'Dutch', 'Korea', 'flingy', 'gagging', 'Want', 'want', 'lighning', 'Eggs', 'takin', 'pigeons', 'Silverman', 'dingle', 'helping', 'Yaaahhh', 'Soo', \
             'because', 'signed', 'horriable', 'Went', 'myself', 'Cats', 'waterbed', 'term', 'that', 'Indian', 'scaring', 'cantelope', 'facing', 'with', 'jammies', 'Once', 'boobie', 'negative', 'any', 'geeky', 'figured', 'Penis', \
             'finding', 'be_okay', 'concept', 'hiring', 'thats', 'Ted', 'been', 'Leroy', 'Cocoon', 'Drops', 'for', 'cookies', 'hearing', 'thoughts', 'Tommy', 'done', 'schwang', 'quitters', 'instance','name', \
             'could_could', 'Bates', 'Oowww', 'Hombre', 'zillionaire', 'strappy', 'Nails', 'Were', 'raquetball', 'Carol', 'miles', 'facts', 'are_are', 'relationships', 'Yeah_eah_ha', 'Might', 'equator', \
             'It', 'this', 'Some', 'Victor', 'Takes', 'Bowe', 'getting', 'cents', 'Alll', 'positive', 'Remore', 'at', 'Heh', 'blobbies', 'fossils', 'Has', 'Gawd', 'other', 'anything', \
             'blocking', 'bought', 'Gaelic', 'alrighty', 'Saturn', 'paleontologist', 'Hands', 'Gellers', 'ok', 'lent', 'This', 'July', 'Ashley', 'then', 'stepped', 'Vietnam', 'Mommies', 'Somebody', 'panty', \
             'Otherwise', 'Ones', 'burnin', 'Alrighty', 'Sarandon', 'on', 'banned', 'played', 'cities', 'swordfish', 'wedgies', 'colour', 'parking', 'Tribianni', 'funniest', 'seatbelts', 'wieght', \
             'walked', 'Poulet', 'bisexuals', 'Ahhhh_gaaaahhh', 'Bookbinders', 'that\'s_that', 'vigilante', 'breaking', 'threw', 'barnicles', 'sangria', 'climbing', 'Apes', 'anyone', 'bye_bye', 'dollhouse', 'flan', 'apartments', 'represents', 'compared', 'parlors', 'Celtics', \
             'making', 'Will', 'opthamologists', 'Eastwood', 'along', 'His', 'dues', 'disappeared', 'WHAT', 'Adelman', 'Adelman', 'Kate', 'afternoons', 'Cubans', 'ok', 'Her', 'seemed', 'wait', 'igneous', 'nights', 'Hi', \
             'mine', 'kept', 'floopy', 'doing', 'feelin', 'finally', 'caterer', 'watching', 'Sapien', 'transferred', 'Things', 'handing', 'Because', 'what_what', 'dumpin', 'Fruitflies', 'those_those', 'Oreos', 'gouged', 'laid', \
             'leaving', 'Doctors', 'margaritas', 'Doritos', 'ever', 'mayo', 'ohh', 'retainer', 'honour', 'Sanskrit', 'Smokey', 'Yeah_no', 'kidder', 'name', 'magazines', 'Agamemnon', 'IDNEY', 'neither', 'somebody', \
             'inlcuding', 'planning', 'topic', 'popping', 'Before', 'instead', 'cocktails', 'lasted', 'fallin', 'hows', 'IS', 'ethnic', 'when_when', 'drank', 'recommending', 'drun', 'you_you', 'you_see', 'talked', 'Besides', 'flipping', \
             'flipping', 'K_Rock', 'ran', 'when', 'gotten', 'schmenis', 'Mets', 'Gunther', 'Ryder', 'Then', 'Eeeee', 'theories', 'NOT', 'butt_munch', 'With', 'this', 'ma\'am', 'cancelling', \
             'Canadians', 'ones', 'Though', 'Nooo', 'Listen', 'like', 'Aaahhhh', 'then_then', 'seashores', 'orthodontist', 'planned', 'Name', 'problems', 'Bing-', 'sort', 'violated', 'lasagnas', 'ahh', 'Amy', \
             'Wolves', 'had', 'flipped', 'mache', 'Macaroons', 'Bugs', 'a_sucks', 'ways', 'unconstituional', 'Flan', 'cran_apple', 'Pepponi', 'females', 'What_what', 'sits', 'Tushie', 'Yesss', 'encyclopedias', 'youre', \
             'Relaxi_Taxi', 'Glen', 'have-have', 'opened', 'Mm_hmm', 'Spackel', 'Beatty', 'British', 'why', 'Buttons', 'Kidman', 'Rockefeller', 'no_no', 'vowels', 'blew', 'lipper', 'MY', \
             'Wont', 'or', 'hearts', 'she', 'But', 'Thurman', 'shall', 'come_here', 'Phoe', 'were', 'rejections', 'hamburgers', 'relaxed', 'everbody', 'limerick', 'Poets', \
             'brown_nosing', 'touchet', 'something', 'tarp', 'Both', 'Shall', 'o_okay', 'Should', 'Poughkeepsie', 'op', 'Playthings', 'puppies', 'actors', 'Kaplan', 'about', 'whoa', \
             'quiches', 'cufflinks', 'bookcase', 'bagels', 'dunno', 'wheres', 'clearer', 'Buon', 'neighbour', 'holdin', 'Leetch', 'took', 'her', 'which', 'it', 'Thinkin', 'you', \
             'Um-hum', 'Here', 'happens', 'YOU', 'Bullwinkle', 'Foghorn', 'goin', 'Y\'know', 'seem' ]
for text in tqdm(text_list):
    text_polarity = sentence_sentiment_polarity(text)[0]
    label.append( text_polarity['label'] )
    print(text," : ",  text_polarity['label'] )
