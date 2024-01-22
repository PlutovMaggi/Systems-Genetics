### Systems Genetics Project ###
# submitted by Maggi Plutov 209223155 & Shani Daniel 315231902 #

# using code from assignments 2 + 3 - detailed below

# The code should get as input the genotypes & phenotypes file (from HW2) 

# chosen data sets file - Liver & Blood Stem Cells
# chosen phenptypes - Cocaine response (10 mg/kg ip), vertical activity (rears) in an open field 30-45 min after injection for females [n beam breaks]
#                   - Cocaine response (2 x 10 mg/kg ip), vertical activity (rears) from 30-45 min after second injection in an activity chamber for females [n beam breaks]

# bsc := blood stem cells
import GEOparse
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import math
import scipy.stats
import seaborn as sns
from scipy.stats import norm
from scipy.stats import linregress
import logging


####################### Q2 - gene expression data preprocessing ########################

# handling unnecessary logs messages
def suppress_geoparse_debug():
    # getting the logger for the GEOparse library
    logger = logging.getLogger('GEOparse')

    # setting the logging level to INFO or higher to suppress DEBUG messages
    logger.setLevel(logging.INFO)
    
    # setting the logging level to WARNING or higher to suppress INFO messages
    logger.setLevel(logging.WARNING)


# reading GEO metadata from .soft file downloaded from GEO
def readGEOsymbols(soft_path):
    # suppresing DEBUG messages from the GEOparse library
    suppress_geoparse_debug()
    
    gse = GEOparse.get_GEO(filepath = soft_path)
    gpl_data = gse.gpls[list(gse.gpls.keys())[0]]
    return gpl_data.table


# reading GEO data from .txt file downloaded from GEO
def readGEOvalues(txt_path):
    # reading the txt file into a df, ignoring first (unneeded) rows
    df = pd.read_csv(txt_path, sep='\t', skiprows=35, header = None, low_memory=False)
    # setting the first column as the row index
    df.set_index(0, inplace=True)
    return df


# creating the gene expression matrix and removing certain elements as stated in part 2 of the project
def create_symbols_BXDs_matrix(df_liver_symbols, df_bsc, df_liver_values, df_bsc_values):
    liver_BXDs = df_liver_values.loc['!Sample_characteristics_ch1'].iloc[0]
    liver_BXDs = liver_BXDs.str.split(':').str[1]
    
    df_liver_values.columns = list(liver_BXDs) # setting the BXDs as the header

    bsc_BXDs = df_bsc_values.loc['!Sample_title']
    bsc_BXDs = bsc_BXDs.str.split(' ').str[0]
    bsc_BXDs = bsc_BXDs.str.replace(r'[a-zA-Z]$', '', regex=True) # removing last letter if exists
    
    df_bsc_values.columns = list(bsc_BXDs) # setting the BXDs as the header
    
    liver_merged_df = create_merged_df(df_liver_symbols, df_liver_values)
    bsc_merged_df = create_merged_df(df_bsc, df_bsc_values)
    
    # renaming the 'Symbol' column to 'GENE_SYMBOL'
    bsc_merged_df.rename(columns={'Symbol': 'GENE_SYMBOL'}, inplace=True)
    
    liver_matrix = organize_df(liver_merged_df)
    bsc_matrix = organize_df(bsc_merged_df)
    
    return (liver_matrix, bsc_matrix)


# merging the annotation df (metadata) with the data df to one df 
def create_merged_df(annotation_df, data_df):
    merged_df = annotation_df.merge(data_df, left_on='ID', right_index=True)
    return merged_df


# returns a pre-processed gene expression matrix
def organize_df(df):
    # filtering unnecessary columns 
    gene_symbol_col = df['GENE_SYMBOL']
    bxd_cols = df.filter(like='BXD') # columns that are not BXD strains are filtered out
    
    # concatenating the filtered columns
    df = pd.concat([gene_symbol_col, bxd_cols], axis=1)
    
    # removing rows with no gene identifier
    df.dropna(subset=['GENE_SYMBOL'], inplace=True)
    
    # setting the gene_symbol as index column 
    df.set_index('GENE_SYMBOL', inplace=True)
    
    # removing unnecessary spaces from column names
    df.columns = [col.replace(' ', '') for col in df.columns]
    
    df = df.astype(float)
    
    # keeping only unique strain names columns (as done in EX3) by calculating average of the common cols
    df = unique_strains_average(df)
    
    # removing rows with low maximal value
    max_values = df.max(axis=1)
    threshold_max = np.percentile(max_values, 50) # removing rows that are in the 50% lowest max values
    #print("the threshold calculated for the maximal values is - " + str(threshold_max))
    max_df = df[max_values >= threshold_max]
    
    # selecting only numeric columns (in case)
    numeric_cols = df.select_dtypes(include=['number'])
    
    # removing rows with low variance
    variances = numeric_cols.var(axis=1)
    threshold_variance = np.percentile(variances, 50) # removing rows that are in the 50% lowest variances
    #print("the threshold calculated for the variances is - " + str(threshold_variance))
    var_df = df[variances >= threshold_variance]
    
    # keeping only rows that are in the top 5% highest maximal value and variance
    common_indices = max_df.index.intersection(var_df.index) 
    df = max_df.loc[common_indices]
    
    # calculating average of rows with the same gene symbol and creating 1 row per symbol
    df = df.groupby('GENE_SYMBOL').mean()

    return df
  
  
# keeps only unique strain names columns (as done in EX3) by calculating average of the common cols
def unique_strains_average(df):
    new_df = pd.DataFrame(index=df.index) # create a new processed df
    unique_strain_names = set(df.columns)
    
    for strain in unique_strain_names:
        # getting the columns for the current strain
        strain_columns = [col for col in df.columns if col == strain]
    
        # calculating the average for the current strain individuals
        averaged_values = df[strain_columns].mean(axis=1)
    
        # assigning the averaged values to a new column in the processed_df 
        new_df[strain] = averaged_values
        
    return new_df
    

# using only representative genomic loci 
# filtering loci that have exactly the same information as neighboring loci across the BXD strains under study
def genotypes_preprocessing(genotypes_df):
    # identifying neighboring loci with the same information
    representative_SNPS = []
    
    for i in range(0, len(genotypes_df)-1):
        curr_loci = genotypes_df.iloc[i, 4:]
        next_loci = genotypes_df.iloc[i+1, 4:]
        
        if not curr_loci.equals(next_loci):
            representative_SNPS.append(genotypes_df.loc[i])
    
    representative_SNPS.append(genotypes_df.loc[len(genotypes_df)-1])
    
    # creating a new DataFrame with only representative genomic loci
    SNP_df = pd.DataFrame(representative_SNPS)
    
    # removing columns with "None" values
    SNP_df = SNP_df.dropna(axis=1, how='any')
       
    return SNP_df

####################### Q3 - eQTL analysis ########################

# using linear regression to compute the p-value of each SNP-gene pair in a df
def regression_model(genotypes_FileName, gene_df):
    genotypes_df = pd.read_excel(genotypes_FileName, skiprows=[0]).replace({'B':0,'b':0,'D':1,'d':1,'H':'H','h':'H','U':None})
    # getting the common column names (common BXDs) for both genes dataset and genotypes
    common_columns = [col for col in gene_df.columns if col in genotypes_df.columns]
    
    # filtering the data frames to keep only the common columns 
    genotypes_df = genotypes_df[genotypes_df.columns[:4].to_list() + common_columns]
    gene_df = gene_df[common_columns]
    
    SNP_df = genotypes_preprocessing(genotypes_df)
    
    # creating an empty data frame to store the p values - rows are SNPs and columns are genes
    pval_df = pd.DataFrame(columns = ['Locus'] + list(gene_df.index))

    for SNP_id, SNP_row in SNP_df.iterrows():
        new_row = [SNP_row['Locus']]
        SNP = np.array(SNP_row.iloc[4:]) # getting only BXD values 
        # removing hetrozygotes
        H_indices = np.where(SNP == 'H')
        H_mask = np.ones(len(SNP), dtype=bool)
        H_mask[H_indices] = False
        SNP = SNP[H_mask] 
        for gene_id, gene_row in gene_df.iterrows():
            gene = np.array(gene_row.iloc[:]) # getting BXD values
            gene = gene[H_mask]
            p_value = compute_p_val(SNP, gene) # getting the current SNP, gene p-value
            new_row.append(p_value) 
        pval_df.loc[len(pval_df)] = new_row
    
    SNP_df = SNP_df.reset_index(drop=True)
    
    # returning P-values data frame
    return pval_df, SNP_df
 
    
# calculates P-value for given SNP & gene values 
def compute_p_val(SNP, gene_or_pheno):
    x = np.array(SNP)
    x = x.astype(float)

    y = np.array(gene_or_pheno)
    y = y.astype(float)

    slope, intercept, r_value, p_val, std_err = linregress(x, y)
    
    return p_val


# applies correction on given P-values by FDR := Benjamini-Hochberg correction - for all the df    
def fdr_correction_eQTLs(pval_df):
    # creating a new Data frame that holds all significant values after correction
    sig_df = pval_df.copy()
    alpha = 0.05 
    
    # applying correction
    columns = sig_df.columns[1:]
    
    # converting df into a 1D array
    sig_data = sig_df[columns].values.flatten()
    
    sig_df_values = fdrcorrection(sig_data)[1]

    # reshaping the corrected p-values back into the original DataFrame shape
    num_rows, num_columns = sig_df.shape
    sig_df_values = sig_df_values.reshape((num_rows, num_columns - 1))
    
    sig_df[columns] = sig_df_values

    # significant eQTL's are converted to 1 (else - 0)
    sig_df[columns] = sig_df[columns].applymap(lambda x: 0 if (float(x) > alpha) else 1)
    
    # excluding genes with weak association to all SNPs
    sig_df = sig_df.loc[:, (sig_df != 0).any(axis=0)]
    
    # returning significant P-values data frame
    return sig_df 


# gets p-values df and returns a df where 1 is significant eQTL and 0 is not, after correction
def compute_significant_eQTLs(pval_df):
    sig_df = fdr_correction_eQTLs(pval_df)
    sig_df.set_index('Locus', inplace=True)
    return sig_df


# plotting the number of genes associated with each eQTL across the genome
def association_plot(name, sig_df, SNP_df):
    Mbp = 10**6 

    eQTL_df = pd.DataFrame(columns=['eQTL','chromosome', 'loc', 'number of genes'])

    # creating dataFrame for plot 
    for SNP_id, SNP_row in sig_df.iterrows():
        eQTL = SNP_id
        eQTL_row = SNP_df[SNP_df['Locus'] == eQTL]
        # getting SNP location
        location = SNP_df.loc[SNP_df['Locus'] == eQTL,'Build37_position'].iloc[0]
        chromosome = int(eQTL_row['Chr_Build37'].values[0])
        num_of_genes = (SNP_row == 1).sum()
        new_row = {'eQTL':eQTL,'chromosome':chromosome, 'loc': location/Mbp, 'number of genes': num_of_genes}
        eQTL_df = pd.concat([eQTL_df,pd.DataFrame([new_row])],ignore_index=True)
    
    # association plot by location across whole genome
    genome_association(name, eQTL_df)
    
    # association plot by location within each chromosome
    chromosome_association(name, eQTL_df)
    
    # grouping the DataFrame by 'chromosome' and sum the 'number of genes'
    genes_per_chromosome = eQTL_df.groupby('chromosome')['number of genes'].sum()

    print(genes_per_chromosome) 


# plotting association plot by location across whole genome
def genome_association(name, eQTL_df):
    eQTL_df = eQTL_df.sort_values(by = ['chromosome','loc'])

    # positions are taken into considiration by commulative sum
    eQTL_df['loc'] = eQTL_df['loc'].cumsum()

    # defining plot
    plt.figure(figsize=(16, 8))
    df = eQTL_df.groupby('chromosome')
    
    # plotting each data point with different colors based on chromosome
    for chromosome, data in df:
        plt.scatter(data['loc'],data['number of genes'],s=20)
    
    eQTL_df['chromosome'] = pd.to_numeric(eQTL_df['chromosome'], errors='coerce')
    eQTL_df['loc'] = pd.to_numeric(eQTL_df['loc'], errors='coerce')
    plt.xticks(eQTL_df.groupby('chromosome')['loc'].median(), labels=eQTL_df['chromosome'].unique())
    
    # setting plot title and axis labels
    plt.title('Number of genes associated with each eQTL (whole genome) in %s' % name)
    plt.xlabel('Chromosome')
    plt.ylabel('number of genes') 
    plt.show()


# plotting association plot by location within each chromosome
def chromosome_association(name, eQTL_df):
    # defining plot
    plt.figure(figsize=(16, 8))
    
    # creating a bar plot
    plt.bar(eQTL_df['loc'], eQTL_df['number of genes'], color ='gray')

    # adding the chromosome number on top of bars to significant values (>=20)
    for eQTL_id, eQTL_row in eQTL_df.iterrows():
        if eQTL_row['number of genes'] >= 20:
            plt.text(eQTL_row['loc'], eQTL_row['number of genes'], str(eQTL_row['chromosome']), ha='center', va='bottom', fontweight='bold', fontsize=12)
            
    # adding a horizontal line at y=20
    plt.axhline(y=20, color='red', linestyle='--')

    #set plot title and axis labels
    plt.title('Number of genes associated with each eQTL (each chromosome) in %s' % name)
    plt.xlabel('Marker position (Mbp)')
    plt.ylabel('number of genes') 
    plt.show()


#define each eQTL by cis-acting or trans-acting 
#input - DataFrame to check, sig_flag = 1 => check only significant values
def acting_type(SNP_df, sig_df, pval_df, df_liver_symbols, sig_flag): 
    trans_pval=[]
    cis_pval=[]
    significant_cols = sig_df.iloc[:, 1:]
    
    for SNP_id, SNP_row in sig_df.iterrows():
        locus = SNP_id
        if (sig_flag == 1): #check only significant columns
            significant_cols = SNP_row[SNP_row == 1].index.tolist()
        for gene in significant_cols:
            pair_type = cis_trans(locus, gene, df_liver_symbols, SNP_df)
            
            if (pair_type == 0): #cis acting
                cis_pval.append(-1*np.log10(pval_df.loc[pval_df['Locus']==locus,gene].values[0])) #add-log(p_val)
                
            if (pair_type == 1): #trans acting
                trans_pval.append(-1*np.log10(pval_df.loc[pval_df['Locus']==locus,gene].values[0])) #add-log(p_val)
        
    return trans_pval,cis_pval
 

def add_parsed_location_data(liver_df):
    liver_df['location'] = liver_df['CHROMOSOMAL_LOCATION'].apply(parse_chromosomal_location)
    
    return liver_df
        

# function to get the liver's genes location
def parse_chromosomal_location(location):
    if pd.notna(location) and location.startswith('chr'):
        parts = location.split(':')
        chromosome = parts[0][3:]
        positions = parts[1].split('-')
        start_position = int(positions[0])
        end_position = int(positions[1])
        return [chromosome, start_position, end_position]
    else:
        return None
    
#returns (0)-cis, (1)-trans, (-1)-undefined
def cis_trans(SNP, gene, liver_df, SNP_df):
    Mbp = 10**6 
    
    #get SNP location
    try: 
        SNP = float(SNP)
    except ValueError:
        pass
        
    SNP_loc = SNP_df.loc[SNP_df['Locus'] == SNP,'Build37_position'].iloc[0]
    SNP_chr = SNP_df.loc[SNP_df['Locus'] == SNP,'Chr_Build37'].iloc[0]

    #get gene location
    try:
        gene_chr, start, end  = liver_df.loc[liver_df['GENE_SYMBOL'] == gene,'location'].iloc[0]
    except TypeError: # no location info
        return -1
    if gene_chr.isdigit() is False:
        return -1
    
    #define each (SNP,gene) pair by cis-acting or trans-acting (cis := 2Mbp or less)
    if (str(SNP_chr) == str(gene_chr)) & (start - 2*Mbp <= SNP_loc <= end + 2*Mbp):
        return 0 #cis-acting
    return 1 #trans-acting


#Q3 in HW3 
# distribution of association P-values scans between cis-associated genes and trans-associated genes.
def distribution_plot(SNP_df, sig_df, pval_df, df_liver_symbols):
    # collect data for significant / non-significant eQTL's
    trans_pval_sig, cis_pval_sig = acting_type(SNP_df, sig_df, pval_df, df_liver_symbols, 1)
    trans_pval, cis_pval = acting_type(SNP_df, sig_df, pval_df, df_liver_symbols, 0)

    # define plot
    plt.figure(figsize=(16, 8))
    # create density plot of significant data
    sns.kdeplot(cis_pval_sig,color = 'blue', label = 'Cis')
    sns.kdeplot(trans_pval_sig, color = 'red', label = 'Trans')
    plt.title('significant distribution plot')
    plt.xlabel('-log(P-value)')
    plt.ylabel('Density') 
    # add legend
    plt.legend()
    # show the plot
    plt.show()
    
    # define plot
    plt.figure(figsize=(16, 8))
    # create density plot of all data
    sns.kdeplot(cis_pval,color = 'blue', label = 'Cis')
    sns.kdeplot(trans_pval, color = 'red', label = 'Trans')
    plt.title('non-significant distribution plot')
    plt.xlabel('-log(P-value)')
    plt.ylabel('Density') 
    # add legend
    plt.legend()
    # show the plot
    plt.show()
 
#Q4 in HW3
# scatter plot visualization (as seen in lecture)
#indicate trans-acting hotspots and clarity of cis-acting eQTLs within it
#location are inside each chromosome - corrected locations need to be considered
def hotspot_plot(SNP_df,sig_df,liver_df):
    Mbp = 10**6
    
    undefined_count = 0
    eQTL_df = pd.DataFrame(columns=['loc_marker','loc_gene','type','eQTL_chromosome'])
    for SNP_id, SNP_row in sig_df.iterrows():
        SNP = SNP_id
        significant_cols = SNP_row[SNP_row == 1].index.tolist()
        #get SNP location
        try: 
            SNP = float(SNP)
        except ValueError:
            pass
        marker_loc = SNP_df.loc[SNP_df['Locus'] == SNP,'Build37_position'].iloc[0]
        
        for gene in significant_cols:
            try:
                chromosome, start, end  = liver_df.loc[liver_df['GENE_SYMBOL'] == gene,'location'].iloc[0]
            except TypeError:
                undefined_count += 1
                continue
            
            if chromosome.isdigit() is False:
                continue
            
            start = int(start)
            end = int(end)
            gene_loc = (start + end) / 2 #take the mean of both locations
            #add new row
            eQTL_type = 'trans' if cis_trans(SNP, gene, liver_df, SNP_df) else 'cis'
            
            new_row = {'loc_marker':marker_loc/Mbp,'loc_gene':gene_loc/Mbp,'type':eQTL_type, 'eQTL_chromosome': chromosome}
            eQTL_df = pd.concat([eQTL_df,pd.DataFrame([new_row])],ignore_index=True)
    
    #show location relative to each chromosome
    chromosome_hotspot(eQTL_df)
    
    #show locations across whole genome
    genome_hotspot(eQTL_df)
    
    return undefined_count

 
#with chromosome location correction - for whole genome
def genome_hotspot(eQTL_df):
    # Apply the conversion function to the 'Chromosome' column
    eQTL_df['eQTL_chromosome'] = eQTL_df['eQTL_chromosome'].apply(convert_chromosome_to_numeric)
    #correct positions by relative distances of each chromosome
    eQTL_df = eQTL_df.sort_values(by = ['eQTL_chromosome','loc_marker'])
    #find the maximum value across all 'loc_marker' and 'loc_gene'
    max_value = eQTL_df[['loc_marker', 'loc_gene']].max().max()
    #update the eQTL_df DataFrame with the maximum values

    eQTL_df['loc_marker'] = eQTL_df['loc_marker'] + max_value * (eQTL_df['eQTL_chromosome'] - 1)
    eQTL_df['loc_gene'] = eQTL_df['loc_gene'] +  max_value * (eQTL_df['eQTL_chromosome'] - 1)

    #define plot
    plt.figure(figsize=(16, 8))
    
    #plotting each data point with different colors based on its type - cis or trans
    cis_data = eQTL_df.loc[eQTL_df['type'] == 'cis']
    trans_data = eQTL_df.loc[eQTL_df['type'] == 'trans']
    plt.scatter(cis_data['loc_marker'], cis_data['loc_gene'],s=30,label = 'Cis')
    plt.scatter(trans_data['loc_marker'], trans_data['loc_gene'], s=5, label = 'Trans')
    
    eQTL_df['eQTL_chromosome'] = pd.to_numeric(eQTL_df['eQTL_chromosome'], errors='coerce')
    eQTL_df['loc_marker'] = pd.to_numeric(eQTL_df['loc_marker'], errors='coerce')
    plt.xticks(eQTL_df.groupby('eQTL_chromosome')['loc_marker'].median(), labels=eQTL_df['eQTL_chromosome'].unique())
    plt.yticks(eQTL_df.groupby('eQTL_chromosome')['loc_marker'].median(), labels=eQTL_df['eQTL_chromosome'].unique())
    
    # Add y=x line
    plt.plot([100,4000], [100,4000], color='gray', linestyle='--', linewidth=1) 
    
    #set plot title and axis labels
    plt.title('Cis & Trans Hotspots (whole genome)')
    plt.xlabel('Chromosome')
    plt.ylabel('Gene position')
    
    #add legend
    plt.legend()
    
    #show the plot
    plt.show()
    
    
#without chromosome location correction - for each chromosome
def chromosome_hotspot(eQTL_df):
    #define plot
    plt.figure(figsize=(16, 8))
    
    #plotting each data point with different colors based on its type - cis or trans
    cis_data = eQTL_df.loc[eQTL_df['type'] == 'cis']
    trans_data = eQTL_df.loc[eQTL_df['type'] == 'trans']
    plt.scatter(cis_data['loc_marker'], cis_data['loc_gene'],s=30,label = 'Cis')
    plt.scatter(trans_data['loc_marker'], trans_data['loc_gene'], s=5, label = 'Trans')
    
    # Add y=x line
    plt.plot([0,200], [0,200], color='gray', linestyle='--', linewidth=1) 
    
    #set plot title and axis labels
    plt.title('Cis & Trans Hotspots (each chromosome)')
    plt.xlabel('Marker position (Mbp)')
    plt.ylabel('Gene position (Mbp)')
    
    #add legend
    plt.legend()
    
    #show the plot
    plt.show()
    
    


def convert_chromosome_to_numeric(chromosome_value):
    return int(chromosome_value)
   

####################### Q4 - QTL analysis ########################

# getting a df of the relevant phenotypes
def phenotypes_preprocessing(phenotypes_FileName, chosen_phenotypes):
    pheno_df = pd.read_excel(phenotypes_FileName)
    phenotypes = []
    # appending our chosen phenotypes
    for phenotype in chosen_phenotypes:
        phenotypes.append(phenotype)
    
    # filtering the DataFrame to keep only rows of our chosen phenotypes
    pheno_df = pheno_df[pheno_df['Phenotype'].isin(phenotypes)]
    return pheno_df
 
    
# computing p-values for each relevant phenotype with all SNPs
def pheno_geno_pvals(genotypes_FileName, pheno_df):
    genotypes_df = pd.read_excel(genotypes_FileName, skiprows=[0]).replace({'B':0,'b':0,'D':1,'d':1,'H':'H','h':'H','U':None})
    SNP_df = genotypes_preprocessing(genotypes_df)
    
    rows_index = list(SNP_df['Locus']) # rows names are the SNPs Locus
    
    res_df = pd.DataFrame(index = rows_index)
    for index, row in pheno_df.iterrows():
        pvals = compute_each_pheno_pvals(row, SNP_df)
        new_column_df = pd.DataFrame({row['Phenotype']: pvals}, index=rows_index)
        res_df = pd.concat([res_df, new_column_df], axis = 1)
    return res_df


# computing p values for all SNPs with specific phenotype
def compute_each_pheno_pvals(phenotype_row, SNP_df):
    phenotype = phenotype_row["BXD1" : "BXD100"] # choosing columns that may contain relevant data
    phenotype = list(phenotype)
    phenotype = np.array([str(item) for item in phenotype]) # ordered phenotype values list
    
    pvals = []
    for index, row in SNP_df.iterrows():
        geno_row = row["BXD1" : "BXD100"] # choosing columns that may contain relevant data
        pval = compute_p_val_GWAS(phenotype, geno_row)
        pvals.append(pval)
    
    return pvals
   
    
# computing p-value for certain phenotype & genotype pair   
def compute_p_val_GWAS(phenotype, geno_row): 
    geno_row = np.array(geno_row)
    no_info_indices = np.where(phenotype == 'nan')
    # creating a boolean mask to identify the indices to delete
    mask = np.ones(len(phenotype), dtype=bool)
    mask[no_info_indices] = False
    
    # using boolean indexing to delete the elements
    phenotype = phenotype[mask]
    geno_row = geno_row[mask] # relevant alleles we have information on
    
    H_indices = np.where(geno_row == 'H')
    H_mask = np.ones(len(phenotype), dtype=bool)
    H_mask[H_indices] = False
    
    # ignoring heterozygotes
    phenotype = phenotype[H_mask]
    geno_row = geno_row[H_mask] 
    
    # converting the array elements to float
    phenotype = phenotype.astype(float)
    
    return compute_p_val(geno_row, phenotype)


# applies correction on given P-values by FDR := Benjamini-Hochberg correction - for each column  
def fdr_correction_QTLs(pval_df):
    # creating a new Data frame that holds all significant values after correction
    sig_df = pval_df.copy()
    alpha = 0.05 
    
    # applying correction
    columns = sig_df.columns[:]
    sig_df[columns] = pval_df[columns].apply(lambda column: multipletests(column, method = "fdr_bh")[1])
    
    # significant eQTL's are converted to 1 (else - 0)
    sig_df[columns] = sig_df[columns].applymap(lambda x: 0 if (float(x) > alpha) else 1) 
    
    # returning significant P-values data frame
    return sig_df


# gets genotypes file name and phenotypes df and returns a df where 1 is significant eQTL and 0 is not, after correction
def compute_significant_QTLs(genotypes_FileName, pheno_df):
    pval_df = pheno_geno_pvals(genotypes_FileName, pheno_df)
    sig_df = fdr_correction_QTLs(pval_df)
    return sig_df


def create_manhattan_plot(SNP_df, pval_df, phenotype_name):  
    positions = SNP_df['Build37_position'] # the positions column (Build37_position)
    chromosomes = SNP_df['Chr_Build37'] # the chromosomes column (Chr_Build37)
    positions = np.array(positions[:]) # all positions in order of the file without header
    chromosomes = np.array(chromosomes[:]) # all chromosomess in order of the file without header

    p_values = -np.log10(np.array(pval_df[phenotype_name]))
    
    plt.figure(figsize=(14, 8))
    
    # creating dataframe with needed data
    df = pd.DataFrame({'pos' : positions,
    'p_value' : p_values,
    'chromosome' : chromosomes})
    
    df['positions_sum'] = df['pos'].cumsum() # for x axis labels and relative positioning
    
    chrom_df = df.groupby('chromosome') # for correct coloring 
    
    for chrom, data in chrom_df:
        plt.scatter(data['positions_sum'],data['p_value']) # scattering in respect to position
        
    plt.xticks(df.groupby('chromosome')['positions_sum'].mean(), labels=[i+1 for i in range(20)]) # x labels
    plt.title('Manhattan Plot - %s' % phenotype_name)
    plt.xlabel('Chromosome')
    plt.ylabel('- Log p-value') 
    plt.show() # shows the plot


#######################  Q5 - combine results ####################### 

def limited_GWAS_pvals_separately(phenotype_pval_df, sig_df):
    # removing rows with only 0s - variants (SNPs) that are associated to at least one expression trait
    sig_df_filtered = sig_df[(sig_df != 0).any(axis=1)]
    relevant_SNPs = sig_df_filtered.index.to_list()

    limited_sig_df = get_limited_sig_df(phenotype_pval_df, relevant_SNPs)
    return limited_sig_df  


def limited_GWAS_pvals_union(phenotype_pval_df, liver_sig_df, bsc_sig_df):
    # removing rows with only 0s - variants (SNPs) that are associated to at least one expression trait
    liver_sig_df_filtered = liver_sig_df[(liver_sig_df != 0).any(axis=1)]
    bsc_sig_df_filtered = bsc_sig_df[(bsc_sig_df != 0).any(axis=1)]
    relevant_SNPs = liver_sig_df_filtered.index.to_list() + bsc_sig_df_filtered.index.to_list()
    relevant_SNPs = list(set(relevant_SNPs))

    limited_sig_df = get_limited_sig_df(phenotype_pval_df, relevant_SNPs)
    return limited_sig_df  


def limited_GWAS_pvals_intersection(phenotype_pval_df, liver_sig_df, bsc_sig_df):
    # removing rows with only 0s - variants (SNPs) that are associated to at least one expression trait
    liver_sig_df_filtered = liver_sig_df[(liver_sig_df != 0).any(axis=1)]
    bsc_sig_df_filtered = bsc_sig_df[(bsc_sig_df != 0).any(axis=1)]
    relevant_SNPs = [snp for snp in liver_sig_df_filtered.index.to_list() if snp in bsc_sig_df_filtered.index.to_list()]

    limited_sig_df = get_limited_sig_df(phenotype_pval_df, relevant_SNPs)
    return limited_sig_df  


def get_limited_sig_df(phenotype_pval_df, relevant_SNPs):
    for i in range(len(relevant_SNPs)):
        try:  
            relevant_SNPs[i] = int(float(relevant_SNPs[i]))
        except ValueError:
            pass
        
    pval_df = phenotype_pval_df.loc[relevant_SNPs]
    limited_sig_df = fdr_correction_QTLs(pval_df)
    return limited_sig_df
    
    
def get_combined_data(gene_df, pheno_df):
    # filter out rows with all zeros
    filtered_gene = gene_df[(gene_df != 0).any(axis=1)]
    filtered_pheno = pheno_df[(pheno_df != 0).any(axis=1)]
    
    # get significant snp names
    sig_gene = filtered_gene.index.to_list()
    sig_pheno =filtered_pheno.index.to_list()

    common_values = [locus for locus in sig_pheno if locus in sig_gene]
    
    combined_data_pheno = pheno_df.loc[common_values]
    combined_data_pheno = combined_data_pheno.loc[:, (combined_data_pheno != 0).any(axis=0)]
    
    combined_data_gene = gene_df.loc[common_values]
    combined_data_gene = combined_data_gene.loc[:, (combined_data_gene != 0).any(axis=0)]
    
    return combined_data_gene, combined_data_pheno


def get_triplets_values(combined_data, SNP_df):
    # create an empty DataFrame with two columns
    common_values = pd.DataFrame(columns=['SNP', 'value', 'SNP_chr']) #value := gene or trait
    
    # find indices where the value is 1
    indices = np.argwhere(combined_data.values == 1)    
    for index in indices:
        # get SNP & gene data
        SNP_idx, value_idx = index
        SNP = combined_data.index[SNP_idx]
        value = combined_data.columns[value_idx]
        
        # get SNP location
        SNP_chr = SNP_df.loc[SNP_df['Locus'] == SNP,'Chr_Build37'].iloc[0]

        # add to dataframe  
        df_to_append = pd.DataFrame({'SNP': SNP, 'value': value, 'SNP_chr': SNP_chr}, index=[0])

        common_values = pd.concat([common_values, df_to_append], ignore_index=True)
        
    return common_values
 
   
def create_exact_triplets_df(common_eQTLs, common_QTLs):
    # create an empty DataFrame for trios of SNP, Gene and Trait
    triplets_df = pd.DataFrame(columns=['SNP', 'Gene', 'Trait', 'chromosome'])
    
    for QTL_idx, QTL_row in common_QTLs.iterrows():
        # find eQTL that are located in a nearby genomic position or have the same position
        QTL_SNP = QTL_row['SNP']
        QTL_trait = QTL_row['value']
        QTL_chr = QTL_row['SNP_chr']

        matching_eQTLs = pd.DataFrame({'SNP': QTL_SNP ,
                                     'Gene': common_eQTLs[(common_eQTLs['SNP'] == QTL_SNP)]['value'],
                                     'Trait': QTL_trait,
                                     'chromosome': QTL_chr}) 
        # add values to dataFrame
        triplets_df = pd.concat([triplets_df, matching_eQTLs[['SNP', 'Gene', 'Trait', 'chromosome']]], ignore_index=True)
    
    return triplets_df
    
    
def limited_GWAS_analysis(gene_df, pheno_df, SNP_df):
    # get common SNPs dataframes for gene and phenotype values
    combined_data_gene, combined_data_pheno = get_combined_data(gene_df, pheno_df)
    
    # get triplets of the common SNPs for each of the analysis
    common_eQTLs = get_triplets_values(combined_data_gene, SNP_df)
    common_QTLs = get_triplets_values(combined_data_pheno, SNP_df)
    
    # get triplets dataFrame with the same SNP 
    triplets_df = create_exact_triplets_df(common_eQTLs, common_QTLs)

    return triplets_df


def get_limited_df(liver_sig_eQTLs, bsc_sig_eQTLs, limited_df_liver, limited_df_bsc, SNP_df):
    liver_df = limited_GWAS_analysis(liver_sig_eQTLs, limited_df_liver, SNP_df)
    liver_df.insert(0, 'tissue', 'liver')
    bsc_df = limited_GWAS_analysis(bsc_sig_eQTLs, limited_df_bsc, SNP_df)
    bsc_df.insert(0, 'tissue', 'bsc')
    df = pd.concat([liver_df, bsc_df], ignore_index=True)
    return df


def Q5_analysis(phenotype_pval_df, phenotype_sig_df, liver_sig_eQTLs, bsc_sig_eQTLs, SNP_df):
    limited_pvals_liver = limited_GWAS_pvals_separately(phenotype_pval_df, liver_sig_eQTLs)
    limited_pvals_bsc = limited_GWAS_pvals_separately(phenotype_pval_df, bsc_sig_eQTLs)

    limited_pvals_union = limited_GWAS_pvals_union(phenotype_pval_df, liver_sig_eQTLs, bsc_sig_eQTLs)
    limited_pvals_intersection = limited_GWAS_pvals_intersection(phenotype_pval_df, liver_sig_eQTLs, bsc_sig_eQTLs)
    
    # current triplets - before limitations 
    before_df = get_limited_df(liver_sig_eQTLs, bsc_sig_eQTLs, phenotype_sig_df, phenotype_sig_df, SNP_df)
    
    # triplets after limiting - separately
    separately_df = get_limited_df(liver_sig_eQTLs, bsc_sig_eQTLs, limited_pvals_liver, limited_pvals_bsc, SNP_df)
    
    # triplets after limiting - union
    union_df = get_limited_df(liver_sig_eQTLs, bsc_sig_eQTLs, limited_pvals_union, limited_pvals_union, SNP_df)
    
    # triplets after limiting - intersection
    intersection_df = get_limited_df(liver_sig_eQTLs, bsc_sig_eQTLs, limited_pvals_intersection, limited_pvals_intersection, SNP_df)
    
    return before_df, separately_df, union_df, intersection_df


#######################  Q6 - causality analysis ####################### 

# returns all trios of significant (SNP, gene, trait)
def sig_trios(eQTL_sig_df, QTL_sig_df, SNP_df):
    Mbp = 10**6
    
    # create an empty DataFrame with two columns
    pair_eQTLs = pd.DataFrame(columns=['SNP', 'Gene', 'SNP_chr', 'SNP_loc'])
    pair_QTLs = pd.DataFrame(columns=['SNP', 'Trait', 'SNP_chr', 'SNP_loc'])
    
    # use numpy to find indices where the value is 1
    eQTL_sig_indices = np.argwhere(eQTL_sig_df.values == 1)    
    for index in eQTL_sig_indices:
        # get SNP & gene data
        SNP_idx, gene_idx = index
        SNP = eQTL_sig_df.index[SNP_idx]
        try: 
            SNP = float(SNP)
        except ValueError:
            pass
        gene = eQTL_sig_df.columns[gene_idx]
        # get SNP location
        SNP_chr = SNP_df.loc[SNP_df['Locus'] == SNP,'Chr_Build37'].iloc[0]
        SNP_loc = SNP_df.loc[SNP_df['Locus'] == SNP,'Build37_position'].iloc[0]
        # add to dataframe
        df_to_append = pd.DataFrame({'SNP': SNP, 'Gene': gene, 'SNP_chr': SNP_chr, 'SNP_loc': SNP_loc}, index=[0])
        pair_eQTLs = pd.concat([pair_eQTLs, df_to_append], ignore_index=True)
        
       
    QTL_sig_indices = np.argwhere(QTL_sig_df.values == 1)
    for index in QTL_sig_indices:
        # get SNP & phenotype data
        SNP_idx, pheno_idx = index
        SNP = QTL_sig_df.index[SNP_idx]
        pheno = QTL_sig_df.columns[pheno_idx]
        # get SNP location
        SNP_chr = SNP_df.loc[SNP_df['Locus'] == SNP,'Chr_Build37'].iloc[0]
        SNP_loc = SNP_df.loc[SNP_df['Locus'] == SNP,'Build37_position'].iloc[0]
        # add to dataframe
        df_to_append = pd.DataFrame({'SNP': SNP, 'Trait': pheno, 'SNP_chr': SNP_chr, 'SNP_loc': SNP_loc}, index=[0])
        pair_QTLs = pd.concat([pair_QTLs, df_to_append], ignore_index=True)

    # create an empty DataFrame for trios of SNP, Gene and Trait
    trios_df = pd.DataFrame(columns=['SNP', 'Gene', 'Trait'])
    
    for QTL_idx, QTL_row in pair_QTLs.iterrows():
        # find eQTL that are located in a nearby genomic position or have the same position
        QTL_SNP = QTL_row['SNP']
        QTL_trait = QTL_row['Trait'] 
        QTL_chr = QTL_row['SNP_chr']
        QTL_loc = QTL_row['SNP_loc']
        # define 2Mbp as proximity threshold for QTL and eQTL to have a nearby genomic position
        nearby_eQTLs = pd.DataFrame({'SNP': QTL_SNP ,
                                     'Gene': pair_eQTLs[(pair_eQTLs['SNP_chr'] == QTL_chr) & (abs(pair_eQTLs['SNP_loc'] - QTL_loc) <= 2*Mbp)]['Gene'].unique(),
                                     'Trait': QTL_trait }) 
        # add values to dataFrame
        trios_df = pd.concat([trios_df, nearby_eQTLs[['SNP', 'Gene', 'Trait']]], ignore_index=True)

    return trios_df


def get_trio_df(trio, SNP_df, gene_df, trait_df):
    # get values from each df
    SNP_values = SNP_df.loc[SNP_df['Locus'] == trio[0]].iloc[:, 4:].T.reset_index().dropna() #BXD values for L
    gene_values = gene_df.loc[trio[1]].to_frame().reset_index().dropna() #BXD values for R
    trait_values = trait_df.loc[trait_df['Phenotype'] == trio[2]].iloc[:,7:].T.reset_index().dropna() #BXD values for C
    # merge all data to create all values df
    trio_df = SNP_values.merge(gene_values, on='index', how='inner').merge(trait_values, on='index', how='inner')
    trio_df.columns = ['individual','L', 'R', 'C']
    trio_df.set_index(trio_df.columns[0], inplace=True)
    return trio_df
    
    
# returns the given trio's predicted relations
def causality_test(trio_df):
    # Sort the DataFrame by the 'L' column in ascending order
    trio_df = trio_df.sort_values(by='L')
    
    C = trio_df['C']
    R = trio_df['R']
    
    # Reset the index of the sorted DataFrame
    trio_df.reset_index(drop=True, inplace=True)
    
    # Group the DataFrame by the 'L' column
    grouped_df = trio_df.groupby('L')
    
    means_C = [0,0]
    means_R = [0,0]
    vars_C = [0,0]
    vars_R = [0,0]
    # Calculate mean and variance for 'C' and 'R' in each group
    for key, group in grouped_df:
        means_C[key] = group['C'].mean()
        vars_C[key] = group['C'].var()
        
        means_R[key] = group['R'].mean()
        vars_R[key] = group['R'].var()
    
    P_Ri_given_Li_list = []
    P_Ci_given_Li_list = []
    P_Ci_given_Ri_list = []
    P_Ri_given_Ci_list = []

    L_R_tuples = list(zip(trio_df['L'], trio_df['R'])) # tuples of L value and corresponding R value
    L_C_tuples = list(zip(trio_df['L'], trio_df['C'])) # tuples of L value and corresponding C value
    R_C_tuples = list(zip(trio_df['R'], trio_df['C'])) # tuples of R value and corresponding C value
    
    correlation_coef = scipy.stats.pearsonr(R, C)[0] # computing ρ
    sigma_C = math.sqrt(trio_df['C'].var())
    sigma_R = math.sqrt(trio_df['R'].var())
    
    myu_C = trio_df['C'].mean()
    myu_R = trio_df['R'].mean()
    
    n = len(trio_df)
    for i in range(n):
        P_Ri_given_Li_list.append(compute_P_Xi_given_Li(means_R, vars_R, L_R_tuples, i))
        P_Ci_given_Li_list.append(compute_P_Xi_given_Li(means_C, vars_C, L_C_tuples, i)) 
        
        Ri, Ci = R_C_tuples[i]
        P_Ci_given_Ri_list.append(compute_P_Xi_given_Yi(correlation_coef, myu_C, myu_R, sigma_C, sigma_R, Ci, Ri, i))
        P_Ri_given_Ci_list.append(compute_P_Xi_given_Yi(correlation_coef, myu_R, myu_C, sigma_R, sigma_C, Ri, Ci, i))
        
    
    M1 = M1_likelihood(P_Ri_given_Li_list, P_Ci_given_Ri_list)
    M2 = M2_likelihood(P_Ci_given_Li_list, P_Ri_given_Ci_list)
    M3 = M3_likelihood(P_Ri_given_Li_list, P_Ci_given_Li_list)
    
    
    # creating a dictionary to store model names and values
    models = {'M1': M1, 'M2': M2, 'M3': M3}
    
    # Find the model with the maximum value
    max_M = max(models, key=models.get)
    max_M_likelihood = models[max_M]
    
    # finding the maximum of the other two values (excluding the maximum value)
    other_likelihoods = [value for key, value in models.items() if value != max_M_likelihood]
    max_of_other_likelihoods = max(other_likelihoods)
    
    # calculating LR
    LR = max_M_likelihood/ max_of_other_likelihoods
    
    return max_M, LR


# for computing P(Ri|Ci) or P(Ci|Ri) 
def compute_P_Xi_given_Yi(correlation_coef, myu_x, myu_y, sigma_x, sigma_y, Xi, Yi, i):
    P_Xi_given_Yi = (1/(math.sqrt(2*math.pi*(sigma_x**2)*(1-correlation_coef**2)))) * \
                    math.exp((-(Xi - myu_x - correlation_coef*(sigma_x/sigma_y)*(Yi-myu_y))**2)/ (2*(sigma_x**2)*(1-correlation_coef**2)))

    return P_Xi_given_Yi


def compute_P_Xi_given_Li(means_list, vars_list, L_X_tuples, i): # normal distribution formula  
    Li, Xi = L_X_tuples[i]
    
    deviations = [math.sqrt(var) for var in vars_list]
    P_Xi_given_Li = (1/(math.sqrt(2*math.pi) * deviations[Li])) * \
                    math.exp(-((Xi - means_list[Li])**2)/ (2*vars_list[Li]))
                    
    return P_Xi_given_Li


# L -> R -> C (locus -> gene -> trait)
def M1_likelihood(P_Ri_given_Li_list, P_Ci_given_Ri_list):
    return comp_likelihood(P_Ri_given_Li_list, P_Ci_given_Ri_list)


# L -> C -> R (locus -> trait -> gene)
def M2_likelihood(P_Ci_given_Li_list, P_Ri_given_Ci_list):
    return comp_likelihood(P_Ci_given_Li_list, P_Ri_given_Ci_list)


#    > C   (      > trait)
# L        (locus        )
#    > R   (      > gene )
def M3_likelihood(P_Ri_given_Li_list, P_Ci_given_Li_list):
    return comp_likelihood(P_Ri_given_Li_list, P_Ci_given_Li_list)


def comp_likelihood(individuals_probs1, individuals_probs2):
    P_Li = 0.5
    likelihood = 1
    
    n = len(individuals_probs1)
    for i in range(n):
        likelihood = likelihood * P_Li * individuals_probs1[i] * individuals_probs2[i]
        
    return likelihood


def permutation_test(trio_df, LR):
    LR_values = []
    
    for i in range(10000): #10,000 permutations
        random_trio_df = randomize_trio(trio_df)
        _ , LR_random = causality_test(random_trio_df)
        LR_values.append(LR_random)
    
    #calculate p_value
    LR_values = np.array(LR_values)
    ln_LR_values = np.log(LR_values)
    ln_LR = math.log(LR)
    p_value_ln = plot_permutation_test(ln_LR_values, ln_LR, 'ln(LR)')
    p_value_norm_ln = plot_norm_permutation_test(ln_LR_values, ln_LR, 'ln(LR)')
    
    #return ((p_value, p_value_norm), (p_value_ln, p_value_norm_ln))
    return (p_value_ln, p_value_norm_ln)


def plot_permutation_test(LR_values, LR, label):
    p_value = (np.sum(LR_values >= LR)) / (len(LR_values))
    
    # create density plot of LR values
    plt.figure(figsize=(16, 8))
    sns.kdeplot(LR_values)
    plt.title('Permutations based LR distribution plot')
    plt.xlabel(label)
    plt.ylabel('Density') 
    plt.axvline(LR, color='red', linestyle='--', label=f'P-Value: {p_value:.4f}')
    plt.legend()
    plt.show()
    
    return p_value
    
    
def plot_norm_permutation_test(LR_values, LR, label):
    max_value = max(LR,max(LR_values))
    min_value = min(LR,min(LR_values))
    x = np.linspace(min(min_value,-1 * max_value ), max(max_value,-1*min_value), 1000)
    
    # Calculate the PDF (Probability Density Function) of the normal distribution
    mean = np.mean(LR_values)
    std = np.std(LR_values)
    pdf = norm.pdf(x, mean, std)
    
    # Calculate the cumulative probability (p-value) for the observed_LR
    p_value_norm = 1 - norm.cdf(LR, loc=mean, scale=std)
    
    # Plot the normal distribution curve
    plt.figure(figsize=(16, 8))
    plt.plot(x, pdf, color='blue', linewidth=2)
    plt.title('normalized permutations based LR distribution plot')
    plt.xlabel(label)
    plt.ylabel('Density') 
    plt.axvline(LR, color='red', linestyle='--', label=f'P-Value: {p_value_norm:.4f}')
    plt.legend()
    plt.show()
    
    return p_value_norm
    

def randomize_trio(trio_df):
    # get R( gene) and C (phenotype) columns and randomize it
    R = trio_df['R']
    C = trio_df['C']
    
    random_R = np.random.permutation(R)
    random_C = np.random.permutation(C)
    
    random_trio_df = trio_df.copy()
    random_trio_df['R'] = random_R
    random_trio_df['C'] = random_C
    
    return random_trio_df


def get_triplets_list(gene_sig_df, phenotype_sig_df, SNP_df, chosen_indices):
    triplets = sig_trios(gene_sig_df, phenotype_sig_df, SNP_df)
    tuple_list = [(row['SNP'], row['Gene'], row['Trait']) for index, row in triplets.iterrows()]
    chosen_triplets = [tuple_list[index] for index in chosen_indices]
    return chosen_triplets


def get_triplets_df(chosen_triplets, SNP_df, pheno_df, gene_df, dataset_name):
    chosen_triplets_df = pd.DataFrame(columns=['DataSet','SNP', 'Gene', 'Trait', 'ln(LR)', 'p-value', 'normalized p-value'])
    for trio in chosen_triplets:
        trio_df = get_trio_df(trio, SNP_df, gene_df, pheno_df)
        best_model, LR = causality_test(trio_df)

        ln_p_values = permutation_test(trio_df, LR)

        triplets_row = pd.DataFrame({'DataSet': dataset_name,
                                     'SNP': trio[0] ,
                                     'Gene': trio[1],
                                     'Trait': trio[2],
                                     'ln(LR)': math.log(LR),
                                     'p-value': ln_p_values[0],
                                     'normalized p-value': ln_p_values[1]},
                                    index=[0])
        # add values to dataFrame
        chosen_triplets_df = pd.concat([chosen_triplets_df, triplets_row], ignore_index=True)

    return chosen_triplets_df
        

def Q6_analysis(liver_sig_eQTLs, bsc_sig_eQTLs, phenotype_sig_df, SNP_df, gene_df_liver, gene_df_bsc, pheno_df):
    #chosen triplets indices
    liver_indices = [0,1,2,3,7]
    bsc_indices = [0,1,3,7,14]
    
    chosen_liver_triplets = get_triplets_list(liver_sig_eQTLs, phenotype_sig_df, SNP_df, liver_indices)
    chosen_bsc_triplets = get_triplets_list(bsc_sig_eQTLs, phenotype_sig_df, SNP_df, bsc_indices)
    
    # liver + bsc triplets as dataFrames
    liver_triplets_df = get_triplets_df(chosen_liver_triplets, SNP_df, pheno_df, gene_df_liver, 'Liver')
    bsc_triplets_df = get_triplets_df(chosen_bsc_triplets, SNP_df, pheno_df, gene_df_bsc, 'BSC')
    
    chosen_triplets_df = pd.concat([liver_triplets_df, bsc_triplets_df], ignore_index=True)
 
    return chosen_triplets_df

############################### main function ##############################


def main_function():
    # part 1 - Defining the task #
    
    chosen_phenotypes = ["Cocaine response (10 mg/kg ip), vertical activity (rears) in an open field 30-45 min after injection for females [n beam breaks]",
                       "Cocaine response (2 x 10 mg/kg ip), vertical activity (rears) from 30-45 min after second injection in an activity chamber for females [n beam breaks]"]
    
    
    # part 2 - Gene expression data preprocessing #
    
    genotypes_df = pd.read_excel("genotypes.xls", skiprows=[0]).replace({'B':0,'b':0,'D':1,'d':1,'H':'H','h':'H','U':None})
    SNP_df = genotypes_preprocessing(genotypes_df)

    pheno_df = phenotypes_preprocessing("phenotypes.xls", chosen_phenotypes)

    df_liver_symbols = readGEOsymbols("GSE17522_family.soft")
    df_bsc_symbols = readGEOsymbols("GSE18067_family.soft") # blood stem cells
    df_liver_values = readGEOvalues("GSE17522_series_matrix.txt")
    df_bsc_values = readGEOvalues("GSE18067_series_matrix.txt")
    dfs = create_symbols_BXDs_matrix(df_liver_symbols, df_bsc_symbols, df_liver_values, df_bsc_values)
    gene_df_liver = dfs[0]
    gene_df_bsc = dfs[1]


    # part 3 - eQTL analysis #
    # association test
    liver_pval_df, liver_SNP_df = regression_model("genotypes.xls", gene_df_liver) 
    bsc_pval_df, bsc_SNP_df = regression_model("genotypes.xls", gene_df_bsc) 

    liver_sig_eQTLs = compute_significant_eQTLs(liver_pval_df)
    bsc_sig_eQTLs = compute_significant_eQTLs(bsc_pval_df)
    
    print("total number of significant liver eQTLs: ", liver_sig_eQTLs[liver_sig_eQTLs == 1].sum().sum())
    print("total number of significant bsc eQTLs: ", bsc_sig_eQTLs[bsc_sig_eQTLs == 1].sum().sum())

    print("total number of significant liver genes: ", len(liver_sig_eQTLs.columns))
    print("total number of significant bsc genes: ", len(bsc_sig_eQTLs.columns))
    
    print("total number of significant liver SNPs: ", (liver_sig_eQTLs == 1).any(axis=1).sum())
    print("total number of significant bsc SNPs: ", (bsc_sig_eQTLs == 1).any(axis=1).sum())
    
    # Question 1 from HW3 - cis&trans acting type
    add_parsed_location_data(df_liver_symbols)
    sig_flag = 1
    trans_pval, cis_pval = acting_type(SNP_df, liver_sig_eQTLs, liver_pval_df, df_liver_symbols, sig_flag)
    print("Number of cis significant eQTLs:", len(cis_pval))
    print("Number of trans significant eQTLs:", len(trans_pval))
    
    # Question 2 from HW3 - association plots
    association_plot("liver", liver_sig_eQTLs, liver_SNP_df)
    association_plot("blood stem cells", bsc_sig_eQTLs, bsc_SNP_df)
    
    # Question 3 - distibution plots for P-values - only for liver dataset
    # Uncomment line below to run - relatively cold
    #distribution_plot(SNP_df, liver_sig_eQTLs, liver_pval_df, df_liver_symbols)

    # Question 4 - cis&trans acting hotspots - only for liver dataset
    hotspot_plot(SNP_df, liver_sig_eQTLs, df_liver_symbols)
    

    # part 4 - QTL analysis #
    phenotype_pval_df = pheno_geno_pvals("genotypes.xls", pheno_df)
    phenotype_sig_df = compute_significant_QTLs("genotypes.xls", pheno_df)

    print("total number of significant QTLs: ", phenotype_sig_df[phenotype_sig_df == 1].sum().sum())
    print("total number of significant SNPs in QTL analysis: ", (phenotype_sig_df == 1).any(axis=1).sum())
    

    # create Manhattan plots for each phenotype
    for phenotype in chosen_phenotypes:
        create_manhattan_plot(SNP_df, phenotype_pval_df, phenotype)


    # part 5 - Combine results #
    
    before_df, separately_df, union_df, intersection_df = Q5_analysis(phenotype_pval_df, phenotype_sig_df, liver_sig_eQTLs, bsc_sig_eQTLs, SNP_df)
    
    print("Number of common QTL/eQTLs per dataset (before limitations): ", before_df.groupby('tissue')['SNP'].nunique()) 
    print("Number of common QTL/eQTLs in total (before limitations): ", before_df['SNP'].nunique())
    
    print("Number of common QTL/eQTLs per dataset (separately): ", separately_df.groupby('tissue')['SNP'].nunique()) 
    print("Number of common QTL/eQTLs in total (separately): ", separately_df['SNP'].nunique())
    
    print("Number of common QTL/eQTLs per dataset (union): ", union_df.groupby('tissue')['SNP'].nunique()) 
    print("Number of common QTL/eQTLs in total (union): ", union_df['SNP'].nunique())
    
    print("Number of common QTL/eQTLs per dataset (intersection): ", intersection_df.groupby('tissue')['SNP'].nunique()) 
    print("Number of common QTL/eQTLs in total (intersection): ", intersection_df['SNP'].nunique())

    # part 6 - Causality analysis # 
    chosen_triplets_df = Q6_analysis(liver_sig_eQTLs, bsc_sig_eQTLs, phenotype_sig_df, SNP_df, gene_df_liver, gene_df_bsc, pheno_df)
    print(chosen_triplets_df)
    
    

######################################################

if __name__ == "__main__":
    main_function()