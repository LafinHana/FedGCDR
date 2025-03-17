import json
def set_dataset(args):
    if args.dataset == 'amazon':
        if args.num_domain == 4:
            with open('data/4domains/domain_user.json', 'r') as f:
                domain_user = json.load(f)
                f.close()
            with open('data/4domains/implicit.json', 'r') as f:
                dic = json.load(f)
                f.close()
            domain_names = ['Clothing', 'Books', 'Movies', 'CDs']
            args.num_users = 55518
        elif args.num_domain == 8:
            with open('data/8domains/domain_user.json', 'r') as f:
                domain_user = json.load(f)
                f.close()
            with open('data/8domains/implicit.json', 'r') as f:
                dic = json.load(f)
                f.close()
            domain_names = ['Clothing', 'Books', 'Home', 'Electronics', 'Sports', 'Cell', 'Movies', 'CDs']
            args.num_users = 98347
        else:
            with open('data/16domains/domain_user.json', 'r') as f:
                domain_user = json.load(f)
                f.close()
            with open('data/16domains/implicit.json', 'r') as f:
                dic = json.load(f)
                f.close()
            domain_names = ['Clothing', 'Books', 'Home', 'Electronics', 'Sports', 'Cell', 'Tools', 'CDs', 'Movies', 'Toys',
                            'Automotive', 'Pet', 'Kindle', 'Office', 'Patio', 'Grocery']
            args.num_users = 117672

    if args.dataset == 'douban':
        args.num_users = 2666
        domain_names = ['Book', 'Movie', 'Music']
        upath = 'data/douban_oldver/domain_user.json'
        dpath = 'data/douban_oldver/implicit.json'
        with open(upath, 'r') as f:
            domain_user = json.load(f)
            f.close()
        with open(dpath, 'r') as f:
            dic = json.load(f)
            f.close()

    return domain_user, dic, domain_names