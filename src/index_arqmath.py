################################################################
# index_arqmath.py
#
# PyTerrier-based Python program for indexing ARQMath data
#
# Author:
# R. Zanibbi, April 2022 (CSCI 539, Information Retrieval, RIT)
################################################################

import pyterrier as pt
import pandas as pd
from bs4 import BeautifulSoup as bsoup
import html
import os, glob, sys
import argparse
from tqdm import tqdm
from arqmath_topics_qrels import *
#from run_topics import *

from math_recoding import *
from utils import *

# Constants for indexing
# **Warning: tag names converted to lower case by default in BSoup (e.g., <P> -> <p>)
TAGS_TO_REMOVE = ['p','a','body','html','question','head','title']

TEXT_RETRIEVAL_FIELDS = [ 'title', 'text', 'tags', 'parentno' ]
TEXT_META_FIELDS = ['docno','title', 'text', 'origtext', 'tags', 'votes', 'parentno', 'mathnos' ]
TEXT_META_SIZES = [ 16, 256, 4096, 4096, 128, 8, 20, 20 ]

MATH_RETRIEVAL_FIELDS = [ 'text', 'parentno' ]
MATH_META_FIELDS = [ 'mathno', 'text', 'origtext','docno','parentno']
MATH_META_SIZES = [ 20, 1024, 1024, 20, 20]

EMPTY_DOCS = 0
REL_THRESHOLD = 2  # ARQMath convention, treat '2' in 0-3 scale as 'relevant' for binary relevance metrics

################################################################
# Index creation and properties
################################################################
def remove_tags( soup_node, tag_list ):
    for tag in soup_node( tag_list ):
        tag.unwrap()

def rewrite_math_tags( soup ):
    # Skip span tags without id's (i.e., formulas without identifiers)
    # Skip empty regions.
    formulaTags = [ node for node in soup('span') if node.has_attr('id') ]
    formula_ids = [ node['id'] for node in formulaTags if node.has_attr('id') ]
    
    #dbshow( "soup", soup )
    #dbshow( "soup('span')", soup('span') )
    #dbshow( "formulaTags", formulaTags )
    #pshow( "formula_ids", formula_ids )

    for tag in formulaTags:
        tag.name = 'math'
        del tag['class']
        #del tag['id']

    return ( formulaTags, formula_ids )

def print_formula_record(math_tag, tokenized_formula, docno, parentno ):
    print('\nDOCNO (FORMULA ID):',math_tag['id'],'\nTEXT:',tokenized_formula,'\nORIGTEXT:',math_tag.get_text(),'\nPOSTNO:',docno,
                                    '\nPARENTNO',parentno)

def print_post_record( docno, title_text, modified_post_text, indexed_body, 
        tag_text, all_formula_ids, parentno, votes):
    print('\nDOCNO: ',docno,'\nTITLE: ',title_text,'\nBODY: ',modified_post_text,
        '\nTEXT (SEARCHABLE):',indexed_body,'\nTAGS: ',tag_text,'\nMATHNOS:',all_formula_ids,
        '\nPARENTNO:',parentno,'\nVOTES:',votes)

def generate_XML_post_docs(file_name_list, formula_index=False, debug_out=False ):
    global EMPTY_DOCS

    for file_name in file_name_list:
        print(">> Reading File: ", file_name )
        with open(file_name) as xml_file:
            soup = bsoup(xml_file, 'lxml')
            rows = soup('row')

            for row in tqdm( rows ):
                # Parse post body and title content as HTML & get formulas
                # Document number in collection, user votes
                docno = row['id'] 
                votes = row['score']

                # Parent post for answers ('qpost' for questions)
                parentno = 'qpost'
                if row[ 'posttypeid' ] == '2':  
                    parentno = row['parentid'] 

                # Title formulas - apply soup to recover HTML structure from attribute field value
                title_soup = bsoup( html.unescape( row.get('title','') ), 'lxml' )
                remove_tags( title_soup, TAGS_TO_REMOVE )
                ( title_formulas, title_formula_ids ) = rewrite_math_tags( title_soup )

                # Body formulas and simplification - again, apply soup to construct Tag tree w. bsoup
                body_soup = bsoup( html.unescape( row.get('body','') ), 'lxml' )
                remove_tags( body_soup, TAGS_TO_REMOVE )
                ( body_formulas, formula_ids )= rewrite_math_tags( body_soup )

                # Remove tags that we do not want to search.
        
                # Combine title and body formulas
                all_formulas = title_formulas + body_formulas
                all_formula_ids = title_formula_ids + formula_ids

                if formula_index:
                    ## Formula index entries   
                    #  One output per formula
                    for math_tag in all_formulas:
                        raw_text = math_tag.get_text()
                        tokenized_formula = rewrite_symbols( math_tag.get_text(), latex_symbol_map )

                        # Skip empty formulas
                        if tokenized_formula.isspace():
                            EMPTY_DOCS += 1
                            if debug_out:
                                if raw_text.isspace():
                                    print('!!! WARNING: Empty "text" field for retrieval, formula id: ' + math_tag['id'] )
                                else:
                                    print('!!! WARNING: Non-empty formula tokenized into empty string, formula id: ' + math_tag['id'])
                                ##print_formula_record(math_tag, tokenized_formula, docno, parentno )
                            continue
                        
                        elif debug_out:
                            print_formula_record( math_tag, tokenized_formula, docno, parentno )

                        yield { 'mathno':     math_tag['id'],
                                'text':      tokenized_formula,
                                'origtext':  math_tag.get_text(),
                                'docno':    docno,
                                'parentno' : parentno
                            }
                else:
                    ## Post text index entries ##
                    # Remove formula ids from title and body
                    #for math_tag in all_formulas:
                    #    del math_tag['id']

                    # Generate strings for title, post body, and tags
                    title_text = str( title_soup )
                    modified_post_text = str( body_soup )
                    indexed_body = translate_latex( modified_post_text )
                    tag_text = row.get('tags', '').replace('<','').replace('>',', ').replace('-',' ')

                    # Skip posts with empty content
                    if indexed_body.isspace():
                        EMPTY_DOCS += 1
                        if debug_out:
                            print('!!! WARNING: Empty "text" field for retrieval, post id: ' + docno ) 
                            #print_post_record( docno, title_text, modified_post_text, indexed_body, 
                            #    tag_text, all_formula_ids, parentno, votes)
                        continue

                    elif debug_out:
                        print_post_record( docno, title_text, modified_post_text, indexed_body, 
                            tag_text, all_formula_ids, parentno, votes)

                    # Note: the formula ids are stored in a string currently.
                    # Concatenate post and tag text
                    # NOTE: representation for search is tokenized differently than meta/document index version for viewing hits
                    yield { 'docno' :   docno,
                            'title' :   title_text,
                            'text' :    indexed_body,
                            'origtext': modified_post_text,
                            'tags' :    tag_text,
                            'mathnos' : all_formula_ids,
                            'parentno': parentno,
                            'votes' :   votes
                        }


def create_XML_index( file_list, indexName, token_pipeline="Stopwords,PorterStemmer", formulas=False, debug=False):
    # Storing processed text AND original text in meta index, docs, to support neural reranking with keywords, and 
    # viewing original posts
    ( meta_fields, meta_sizes ) = ( TEXT_META_FIELDS, TEXT_META_SIZES )
    field_names= TEXT_RETRIEVAL_FIELDS

    if formulas:
        ( meta_fields, meta_sizes ) = ( MATH_META_FIELDS, MATH_META_SIZES )
        field_names= MATH_RETRIEVAL_FIELDS

    indexer = pt.IterDictIndexer( 
            indexName, 
            meta=meta_fields,
            meta_lengths=meta_sizes,
            overwrite=True )

    indexer.setProperty( "termpipelines", token_pipeline )
    index_ref = indexer.index( generate_XML_post_docs( file_list, formula_index=formulas, debug_out=debug ), fields=field_names )

    if EMPTY_DOCS > 0:
        count = str( EMPTY_DOCS )
        print("*** WARNING: " + count + " documents/formulas empty before tokenization, and were skipped.")
        print("    Additional documents/formulas may be empty after tokenization (PyTerrier message will report)")
    
    return pt.IndexFactory.of( index_ref )

## Visualization routines

def show_tokens( index ):
    # Show lexicon entries
    for kv in index.getLexicon():
        print("%s :    %s" % (kv.getKey(), kv.getValue().toString()) )    

def show_index_stats( index ):
    print( index.getCollectionStatistics().toString() )

def view_index( indexName, index, view_tokens, view_stats ):
    if view_tokens or view_stats:
        print('\n[ ' + indexName + ': Details ]')
        if view_stats:
            show_index_stats( index )
        if view_tokens:
            print('Lexicon for ' + indexName + ':')
            show_tokens( index )
            print('')

################################################################
# Search engine construction and search
################################################################
def search_engine( index, 
        model, 
        metadata_keys=[], 
        token_pipeline="Stopwords,PorterStemmer" ):
    return pt.BatchRetrieve( index, wmodel=model, 
            properties={ "termpipelines" : token_pipeline }, 
            metadata = metadata_keys )

# Run a single query
def query( engine, query ):
    return engine.search( translate_query( query ) )

# Run a list of queries
def batch_query( engine, query_list ):
    column_names=[ "qid", "query" ]
    
    query_count = len(query_list)
    qid_list = [ str(x) for x in range(1, query_count + 1) ]

    # Map TeX characters and ARQMath-version query ops (e.g., '_pand' -> '+')
    print( query_list )
    rewritten_query_list = translate_qlist( query_list )
    
    query_pairs = list( zip( qid_list, rewritten_query_list ) )
    queries = pd.DataFrame( query_pairs, columns=column_names )

    return engine( queries )

def verbose_hit_summary( result, math_index=False ):

    result.reset_index()
    for ( index, row ) in result.iterrows():
        #print("KEYS: " + str( row.keys() ) )
        print('QUERY (' + row['qid'] + '): ', row['query'])
        score = "{:.2f}".format( row['score'] )
        
        print('RANK:', index, 'Score:', score)
        if not math_index:
            # Post document
            print('Docid:',row['docid'], 'Post-no:', row['docno'], 'Parent-no:',row['parentno'],'Votes:', row['votes'] )
            if row['parentno'] == 'qpost':
                print('QUESTION TITLE:', row['title'])
            else:
                print('ANSWER')
        else:
            # Formula document
            print('Docid:',row['docid'], 'Formula-no:', row['mathno'],  'Post-no (docno):', row['docno'], 'Parent-no:',row['parentno'])

        # Show original text before token mapping
        print('TEXT:\n', row['text'])
        print('ORIGTEXT:\n',row['origtext'])

        # Provide tags, formula id's for posts
        if not math_index:
            print('TAGS:',row['tags'])
            print('FORMULA IDS:',row['mathnos'])
        
        print('')

def show_result( result, field_names=[], show_table=True, show_hits=False, math=False ):
    print("\n__ SEARCH RESULTS _________________________\n")

    if show_table:
        if field_names == []:
            print( result, '\n' )
        else:
            print( result[ field_names ], '\n' )

    if show_hits:
        verbose_hit_summary( result, math_index=math )

def test_retrieval( k, post_index, math_index, model, tokens, debug=False ):

    if post_index != None:
        print("[ Testing post index retrieval ]")
        
        # Return top k results (% k)
        posts_engine = search_engine( post_index, 
                model, 
                metadata_keys=['docno','title', 'text', 'origtext', 'tags', 'votes', 'parentno', 'mathnos' ],
                token_pipeline=tokens ) % k
        
        result = query( posts_engine, '_pand simplified _pand proof' )
        show_result( result, [], show_hits=True )
        # Added 'writing' to test matching tags, 'mean' in title field  for post number '1'
        show_result( batch_query( posts_engine, [
            'simplified proof', 
            'proof', 
            'writing', 
            'mean', 
            'qpost', 
            'proof _pnot qpost' 
            # 'man +TITLE:{intuition}'  # Trouble restricting to fields (?)
            ] ), [], show_hits=True )
    
    if math_index != None:
        print("[ Testing math index retrieval ]")
        
        # Return top k results (% k)
        math_engine = search_engine( math_index, model, ['mathno', 'text', 'origtext','docno', 'parentno' ], token_pipeline=tokens ) % k
        show_result( query( math_engine, '_pand sqrt _pand 2' ), show_hits=True, math=True )
        show_result( batch_query( math_engine, [ 'sqrt 2', '2' ] ), show_hits=True, math=True )
        show_result( batch_query( math_engine, [ 'sqrt 2 _pnot qpost' ] ), show_hits=True, math=True )

    print( 'Test complete.' )


################################################################
# Main program
################################################################
def process_args():
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Indexing tool for ARQMath data.")

    parser.add_argument('xmlFile', help='ARQMath XML file or directory of XML files to index')
    xgroup = parser.add_mutually_exclusive_group(required=False)
    xgroup.add_argument('-m', '--math', help='create only the math index', action="store_true" )
    xgroup.add_argument('-mp', '--mathpost', help='create math and post indices', action="store_true")
    parser.add_argument('-l', '--lexicon', help='show lexicon', action="store_true" )
    parser.add_argument('-s', '--stats', help="show collection statistics", action="store_true" )
    parser.add_argument('-t', '--tokens', help="set tokenization property (none:  no stemming/stopword removal)", 
            default='Stopwords,PorterStemmer' )
    parser.add_argument('-n', '--notest', help="skip retrieval tests after indexing", action="store_true" )
    parser.add_argument('-d', '--debug', help="include debugging outputs", action="store_true" )
    
    args = parser.parse_args()
    return args

def report_results(ndcg_metrics, binarized_metrics, reranking_epic, top_k, prime):
    # Make clear what we're using!
    prime_string = ''

    if prime:
        prime_string = "'"
    print("[[ Evaluation  ]]")
    print(" * Top-k hits evaluated: " + str(top_k))
    print(" * Prime metrics ('): " + str(prime))
    print(" * !! Note that ARQMath uses prime metrics for official scores.")
    print("\nResults for nDCG" + prime_string)
    print("----------------------------------------------------------")
    print(ndcg_metrics)

    print("\nResults for binarized relevance : mAP" + prime_string + ", Precision at 10" + prime_string)
    print("----------------------------------------------------------")
    print(binarized_metrics)

    print("\nResults for EPIC re-ranking relevance : mAP" + prime_string + ", Precision at 10" + prime_string)
    print("----------------------------------------------------------")

    print("\ndone.")

def select_assessed_hits(qrel_df, top_k=1000, prime=True):
    def filter_results(result_df):
        # result_df.drop_duplicates( subset='docno' )  # esp. important for formula retrieval results
        # result_df_cut = result_df.iloc[0 : MAX_HITS ]
        result_df_cut = result_df.loc[result_df['rank'] < top_k]
        out_results = result_df_cut

        # Prime metrics
        # If 'prime' is true, filter by ( qid, docno ) pairs in the qrel file, so that
        # only assessed hits are included.
        if prime:
            keycols = qrel_df[['qid', 'docno']]
            keys = list(keycols.columns.values)
            i1 = result_df_cut.set_index(keys).index
            i2 = qrel_df.set_index(keys).index
            out_results = result_df_cut[i1.isin(i2)]

        return out_results

    return pyterrier.apply.generic(filter_results)

def load_qrels_with_binary(file_name):
    qrels_df = load_qrels(file_name)
    # !! Create thresholded qrels for map' and p'@10
    qrels_thresholded = qrels_df[qrels_df['label'] >= REL_THRESHOLD]

    return (qrels_df, qrels_thresholded)

def load_topics(file_name):
    (num_topics, topics_df) = read_topic_file(file_name)
    queries = topics_df['query']
    qids = topics_df['qid']
    query_df = pt.new.queries(queries, qid=qids)

    return (num_topics, query_df)

def load_index(index_dir, lexicon, stats):
    index_path = index_dir + "/data.properties"
    print("Loading index defined at " + index_path + "...")
    index_ref = pt.IndexRef.of(index_path)
    index = pt.IndexFactory.of(index_ref)

    # If asked, report stats and lexicon
    view_index(index_dir, index, lexicon, stats)

    return index

def main():

    top_k = 5
    prime = True
    weight_model = "BM25"

    # Process arguments
    args = process_args()
    ( indexName, _ ) = os.path.splitext( os.path.basename( args.xmlFile ) )
    # Set pandas display width wider
    pd.set_option('display.max_colwidth', 150)

    if args.tokens == 'none':
        args.tokens = ''

    in_file_list = [ args.xmlFile ]
    if os.path.isdir( args.xmlFile ):
        # Get directory, keep only xml files.
        dir_files = os.listdir( args.xmlFile )
        in_file_list = [ (args.xmlFile + "/" + file).replace("//","/")
                for file in dir_files if file.endswith('.xml') ]

    # Start PyTerrier -- many Java classes unavailable until this is complete
    print('\n>>> Initializing PyTerrier...')
    if not pt.started():
        pt.init()
    import onir_pt
    
    # Collect topics, qrels index
    print("Loading topics (queries)...")
    (num_topics, query_df) = load_topics(args.xmlFile)
    print("    " + str(num_topics) + " topics lodaded.")

    print("Loading qrels...")
    (qrels_df, qrels_thresholded) = load_qrels_with_binary('./ARQMath_Evaluation/qrels_task_1/2021_qrels_task1.tsv') # used to be args.qrelFile

    indexDir = './ARQMath_Collection-post-ptindex'
    print("Loading index defined at " + indexDir + "...")
    index = load_index(indexDir, args.lexicon, args.stats)
    prime_transformer = select_assessed_hits(qrels_df, top_k, prime)
    bm25_engine = search_engine(index, weight_model, TEXT_META_FIELDS, token_pipeline=args.tokens)
    bm25_pipeline = bm25_engine >> prime_transformer

    # experiment.py stuff:
    
    # end experiment.py stuff

    print('\n>>> Indexing...')
    # Initialize indices as non-existent
    post_index = None
    math_index = None
    
    # Post index construction
    # Store post text and ids for formulas in each post in the 'meta' (document) index
    if not args.math or args.mathpost:
        post_index = create_XML_index(
            in_file_list, "./" + indexName + "-post-ptindex", 
            token_pipeline=args.tokens, debug=args.debug )
        view_index( "Post Index", post_index, args.lexicon, args.stats )

    # Formula index construction
    # Store formula text (LaTeX) and formula ids, along with source post id for each formula
    if args.math or args.mathpost:
        math_index = create_XML_index( 
            in_file_list, "./" + indexName + "-math-ptindex", formulas=True, 
            token_pipeline=args.tokens, debug=args.debug )
        view_index( "Math Index", math_index, args.lexicon, args.stats )

    print('>>> Indexing complete.\n')

    print("Ari's thing:")

    lazy_epic = onir_pt.reranker.from_checkpoint('https://macavaney.us/epic.msmarco.tar.gz')

    br = pt.BatchRetrieve(index)

"""
    from sklearn.model_selection import train_test_split
    #train_ds = pt.datasets.get_dataset('./run-topics-2020')
    train_topics, valid_topics = train_test_split('./run-topics-2020', test_size=16, random_state=42) # split into training and validation sets
    # train_topics = query_df

    # Index MS MARCO
    indexer = pt.index.IterDictIndexer('./ARQMath_Collection')
    tr_index_ref = indexer.index('./ARQMath_Collection-post-ptindex', fields=('text',), meta=('docno',))

    pipeline = (pt.BatchRetrieve(tr_index_ref) % 100 # get top 100 results
        >> pt.text.get_text('./ARQMath_Collection-post-ptindex', 'text') # fetch the document text
        >> pt.apply.generic(lambda df: df.rename(columns={'text': 'abstract'})) # rename columns
        >> knrm) # apply neural re-ranker

    pipeline.fit(
        train_topics,
        train_ds.get_qrels(),
        valid_topics,
        train_ds.get_qrels())


"""
"""

    from sklearn.model_selection import train_test_split
    # train_ds = pt.datasets.get_dataset('irds:msmarco-passage/train/medical')
    train_topics, valid_topics = train_test_split('./run-topics-2020', test_size=3, random_state=42) # split into training and validation sets

    dataset = generate_XML_post_docs( in_file_list , True, debug_out=args.debug )

    indexed_epic = onir_pt.indexed_epic.from_checkpoint('https://macavaney.us/epic.msmarco.tar.gz', index_path = './epic')

    index_ref = indexed_epic #.index( dataset )
    # Previously:
    # index_ref = indexed_epic.index( dataset, fields=TEXT_META_FIELDS )
    # indexed_epic.index(index_ref, fields=('TEXT_META_FIELDS'))

    epic_pipeline = (bm25_pipeline >> indexed_epic.reranker())

    ndcg_metrics = pt.Experiment(
        [bm25_pipeline, epic_pipeline],
        query_df,
        qrels_df,
        eval_metrics=["ndcg", "mrt"],
        names=[weight_model],
        save_dir="./",
        save_mode="overwrite"
    )

    binarized_metrics = pt.Experiment(
        [bm25_engine, epic_pipeline],
        query_df,
        qrels_thresholded,
        eval_metrics=["P_10", "map", "mrt"],
        names=[weight_model],
        save_dir="./"
    )

    cutoffs = [10]  #, 50, 100]
    reranking_depth = pt.Experiment(
        [bm25_engine % cutoff >> indexed_epic.reranker() for cutoff in cutoffs],
        query_df,
        qrels_thresholded,
        eval_metrics=["P_10", "map", "ndgc", "mrt"],
        names=[weight_model],
        save_dir="./"
    )

    

    # Report results at the command line.
    report_results(ndcg_metrics, binarized_metrics, reranking_depth, top_k, prime)



    # Retrieval test
    # Top k
    k = 5
    if not args.notest:
        test_retrieval( k, post_index, math_index, 'BM25', args.tokens, debug=args.debug )  

"""  

if __name__ == "__main__":
    main()
