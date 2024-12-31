#!/usr/bin/env python

# @Time    : 2024/9/25 20:22
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : GeneAggr2Set.py
import pandas as pd
from tqdm import tqdm
import anndata
from scipy.sparse import csr_matrix


def get_mapping_gene(mapping, method):
    genes = []
    if (method == 'mean') or (method == 'sum'):
        for gs in mapping['gene_set'].unique():
            gene = gs.split(',')
            if isinstance(gene, str):
                genes.append(gene.strip('(').strip(')').strip())
            else:
                for g in gene:
                    g = g.strip('(').strip(')').strip()
                    genes.append(g)

    else:
        for gs in mapping['gene_set'].unique():
            gs = gs.replace(' and ', ',').replace(' or ', ',')
            gene = gs.split(',')
            if isinstance(gene, str):
                genes.append(gene.strip('(').strip(')').strip())
            else:
                for g in gene:
                    g = g.strip('(').strip(')').strip()
                    genes.append(g)
    return list(set(genes))


def supplement(adata_df, gene_list):
    add_data = pd.DataFrame(0.0, columns=list(set(gene_list) - set(adata_df.columns)), index=adata_df.index, )
    return pd.concat([adata_df, add_data], axis=1)


def min_nonzero(row):
    row_values = row[row != 0].values  # 获取非零值
    if len(row_values) == 0:
        return 0
    else:
        return min(row_values)


def gene_aggr_to_set(adata, mapping, method):
    adata_df = pd.DataFrame(adata.X.toarray(), index=adata.obs.index, columns=adata.var.index)
    mapping = pd.read_csv(mapping, sep='\t')
    mapping_genes = get_mapping_gene(mapping=mapping, method=method)
    print(mapping_genes)
    print(len(mapping_genes))
    adata_df = supplement(adata_df, mapping_genes)

    set_value_in_cells = []
    pbar = tqdm(mapping.iterrows())
    for i, row in pbar:
        reaction = row.reaction
        if method == 'sum':
            gene_set = row.gene_set.split(',')
            gene_set = list(set(gene_set))
            set_value_in_cells.append(adata_df.loc[:, gene_set].sum(axis=1).rename(reaction, inplace=True))
        if method == 'mean':
            gene_set = row.gene_set.split(',')
            gene_set = list(set(gene_set))
            set_value_in_cells.append(adata_df.loc[:, gene_set].mean(axis=1).rename(reaction, inplace=True))
        if method == 'set':
            # 单一关系
            if '(' not in row.gene_set:
                if 'or' in row.gene_set:
                    or_gene_set = row.gene_set.split(' or ')
                    set_value_in_cells.append(adata_df.loc[:, or_gene_set].max(axis=1).rename(reaction, inplace=True))
                elif 'and' in row.gene_set:
                    or_gene_set = row.gene_set.split(' and ')
                    set_value_in_cells.append(adata_df.loc[:, or_gene_set].sum(axis=1).rename(reaction, inplace=True))
                else:
                    or_gene_set = row.gene_set.split('---')
                    set_value_in_cells.append(adata_df.loc[:, or_gene_set].max(axis=1).rename(reaction, inplace=True))

            # 超复杂关系
            elif ('((') in row.gene_set:
                complicated_gene_set_values = []
                row_gene_set = row.gene_set.split(')) and ((')
                for sub_row_gene_set in row_gene_set:
                    if (') and' in sub_row_gene_set) | ('and (' in sub_row_gene_set):
                        and_gene_set = sub_row_gene_set.split(' and ')
                        and_gene_set_values_in_cells = []
                        for ags in and_gene_set:
                            ags = ags.strip('(').strip(')')
                            or_gene_set = ags.split(' or ')
                            if len(or_gene_set) == 1:
                                or_gene_set_values_in_cell = adata_df.loc[:, or_gene_set]
                            else:
                                or_gene_set_values_in_cell = adata_df.loc[:, or_gene_set].max(axis=1)
                            and_gene_set_values_in_cells.append(or_gene_set_values_in_cell)
                        and_gene_set_values_in_cells = pd.concat(and_gene_set_values_in_cells, axis=1)
                        complicated_gene_set_values.append(
                            and_gene_set_values_in_cells.sum(axis=1).rename(reaction, inplace=True))
                    # 复合关系 (A and B) or (C and D) or G
                    elif (') or' in sub_row_gene_set) | ('or (' in sub_row_gene_set):
                        or_gene_set = sub_row_gene_set.split(' or ')
                        or_gene_set_values_in_cells = []
                        for ogs in or_gene_set:
                            ogs = ogs.strip('(').strip(')')
                            and_gene_set = ogs.split(' and ')
                            if len(and_gene_set) == 1:
                                and_gene_set_values_in_cell = adata_df.loc[:, and_gene_set]
                            else:
                                and_gene_set_values_in_cell = adata_df.loc[:, and_gene_set].sum(axis=1)
                            or_gene_set_values_in_cells.append(and_gene_set_values_in_cell)
                        or_gene_set_values_in_cells = pd.concat(or_gene_set_values_in_cells, axis=1)
                        complicated_gene_set_values.append(
                            or_gene_set_values_in_cells.max(axis=1).rename(reaction, inplace=True))

                complicated_gene_set_values = pd.concat(complicated_gene_set_values, axis=1)
                set_value_in_cells.append(complicated_gene_set_values.sum(axis=1).rename(reaction, inplace=True))

            # 复合关系 (A or B) and (C or D) and G
            elif (') and' in row.gene_set) | ('and (' in row.gene_set):
                and_gene_set = row.gene_set.split(' and ')
                and_gene_set_values_in_cells = []
                for ags in and_gene_set:
                    ags = ags.strip('(').strip(')')
                    or_gene_set = ags.split(' or ')
                    if len(or_gene_set) == 1:
                        or_gene_set_values_in_cell = adata_df.loc[:, or_gene_set]
                    else:
                        or_gene_set_values_in_cell = adata_df.loc[:, or_gene_set].max(axis=1)
                    and_gene_set_values_in_cells.append(or_gene_set_values_in_cell)
                and_gene_set_values_in_cells = pd.concat(and_gene_set_values_in_cells, axis=1)
                set_value_in_cells.append(and_gene_set_values_in_cells.sum(axis=1).rename(reaction, inplace=True))
            # 复合关系 (A and B) or (C and D) or G
            elif (') or' in row.gene_set) | ('or (' in row.gene_set):
                or_gene_set = row.gene_set.split(' or ')
                or_gene_set_values_in_cells = []
                for ogs in or_gene_set:
                    ogs = ogs.strip('(').strip(')')
                    and_gene_set = ogs.split(' and ')
                    if len(and_gene_set) == 1:
                        and_gene_set_values_in_cell = adata_df.loc[:, and_gene_set]
                    else:
                        and_gene_set_values_in_cell = adata_df.loc[:, and_gene_set].sum(axis=1)
                    or_gene_set_values_in_cells.append(and_gene_set_values_in_cell)
                or_gene_set_values_in_cells = pd.concat(or_gene_set_values_in_cells, axis=1)
                set_value_in_cells.append(or_gene_set_values_in_cells.max(axis=1).rename(reaction, inplace=True))

        if method == 'set2':
            # 单一关系
            if '(' not in row.gene_set:
                if 'or' in row.gene_set:
                    or_gene_set = row.gene_set.split(' or ')
                    set_value_in_cells.append(adata_df.loc[:, or_gene_set].sum(axis=1).rename(reaction, inplace=True))
                elif 'and' in row.gene_set:
                    or_gene_set = row.gene_set.split(' and ')
                    set_value_in_cells.append(
                        adata_df.loc[:, or_gene_set].apply(min_nonzero, axis=1).rename(reaction, inplace=True))
                else:
                    or_gene_set = row.gene_set.split('---')
                    set_value_in_cells.append(adata_df.loc[:, or_gene_set].sum(axis=1).rename(reaction, inplace=True))

            # 超复杂关系
            elif ('((') in row.gene_set:
                complicated_gene_set_values = []
                row_gene_set = row.gene_set.split(')) and ((')
                for sub_row_gene_set in row_gene_set:
                    if (') and' in sub_row_gene_set) | ('and (' in sub_row_gene_set):
                        and_gene_set = sub_row_gene_set.split(' and ')
                        and_gene_set_values_in_cells = []
                        for ags in and_gene_set:
                            ags = ags.strip('(').strip(')')
                            or_gene_set = ags.split(' or ')
                            if len(or_gene_set) == 1:
                                or_gene_set_values_in_cell = adata_df.loc[:, or_gene_set]
                            else:
                                or_gene_set_values_in_cell = adata_df.loc[:, or_gene_set].sum(axis=1)
                            and_gene_set_values_in_cells.append(or_gene_set_values_in_cell)
                        and_gene_set_values_in_cells = pd.concat(and_gene_set_values_in_cells, axis=1)
                        complicated_gene_set_values.append(
                            and_gene_set_values_in_cells.apply(min_nonzero, axis=1).rename(reaction, inplace=True))
                    # 复合关系 (A and B) or (C and D) or G
                    elif (') or' in sub_row_gene_set) | ('or (' in sub_row_gene_set):
                        or_gene_set = sub_row_gene_set.split(' or ')
                        or_gene_set_values_in_cells = []
                        for ogs in or_gene_set:
                            ogs = ogs.strip('(').strip(')')
                            and_gene_set = ogs.split(' and ')
                            if len(and_gene_set) == 1:
                                and_gene_set_values_in_cell = adata_df.loc[:, and_gene_set]
                            else:
                                and_gene_set_values_in_cell = adata_df.loc[:, and_gene_set].apply(min_nonzero, axis=1)
                            or_gene_set_values_in_cells.append(and_gene_set_values_in_cell)
                        or_gene_set_values_in_cells = pd.concat(or_gene_set_values_in_cells, axis=1)
                        complicated_gene_set_values.append(
                            or_gene_set_values_in_cells.sum(axis=1).rename(reaction, inplace=True))

                complicated_gene_set_values = pd.concat(complicated_gene_set_values, axis=1)
                set_value_in_cells.append(complicated_gene_set_values.apply(min_nonzero, axis=1).rename(reaction, inplace=True))

            # 复合关系 (A or B) and (C or D) and G
            elif (') and' in row.gene_set) | ('and (' in row.gene_set):
                and_gene_set = row.gene_set.split(' and ')
                and_gene_set_values_in_cells = []
                for ags in and_gene_set:
                    ags = ags.strip('(').strip(')')
                    or_gene_set = ags.split(' or ')
                    if len(or_gene_set) == 1:
                        or_gene_set_values_in_cell = adata_df.loc[:, or_gene_set]
                    else:
                        or_gene_set_values_in_cell = adata_df.loc[:, or_gene_set].sum(axis=1)
                    and_gene_set_values_in_cells.append(or_gene_set_values_in_cell)
                and_gene_set_values_in_cells = pd.concat(and_gene_set_values_in_cells, axis=1)
                set_value_in_cells.append(and_gene_set_values_in_cells.apply(min_nonzero, axis=1).rename(reaction, inplace=True))
            # 复合关系 (A and B) or (C and D) or G
            elif (') or' in row.gene_set) | ('or (' in row.gene_set):
                or_gene_set = row.gene_set.split(' or ')
                or_gene_set_values_in_cells = []
                for ogs in or_gene_set:
                    ogs = ogs.strip('(').strip(')')
                    and_gene_set = ogs.split(' and ')
                    if len(and_gene_set) == 1:
                        and_gene_set_values_in_cell = adata_df.loc[:, and_gene_set]
                    else:
                        and_gene_set_values_in_cell = adata_df.loc[:, and_gene_set].apply(min_nonzero, axis=1)
                    or_gene_set_values_in_cells.append(and_gene_set_values_in_cell)
                or_gene_set_values_in_cells = pd.concat(or_gene_set_values_in_cells, axis=1)
                set_value_in_cells.append(or_gene_set_values_in_cells.sum(axis=1).rename(reaction, inplace=True))

        pbar.set_description("Aggregating Genes to Reaction")
    set_value_in_cells = pd.concat(set_value_in_cells, axis=1)
    aggr_adata = anndata.AnnData(csr_matrix(set_value_in_cells.values), obs=adata.obs)
    aggr_adata.var.index = set_value_in_cells.columns
    return aggr_adata


def pick_gene(adata,mapping):
    picked_genes = get_mapping_gene(mapping,method='mean')


def convert():
    g_id = pd.read_csv(r"Z:\Work\Post_COVID19\gene_set\eid_gene.txt", sep='\t')
    mapping = pd.read_csv(r"Z:\Work\Post_COVID19\gene_set\gene_set_dict.pre.tsv", sep='\t', encoding='utf-8')
    mapping.dropna(subset='GENE ASSOCIATION', inplace=True)
    print(g_id)
    print(mapping)
    gs = mapping['GENE ASSOCIATION']
    convert_gs = []
    for gss in tqdm(gs):
        for i, eid in g_id.iterrows():
            gss = gss.replace(eid.NAME, eid['SHORT NAME'])
        convert_gs.append(gss)
        # print(convert_gs)
    new_mapping = pd.DataFrame()
    new_mapping['reaction'] = mapping['ID']
    new_mapping['gene_set'] = convert_gs
    new_mapping.to_csv(r"Z:\Work\Post_COVID19\gene_set\gene_set_dict.3.tsv", sep='\t')

# def convert()
# convert()


# def test():
#     adata = sc.read_h5ad(r"Z:\Work\Post_COVID19\workflow\dataset\all_human.h5ad")
#     print(adata)
#     mapping = "Z:\Work\Post_COVID19\gene_set\gene_set_dict.3.tsv"
#     aggr_data = gene_aggr_to_set(adata, mapping, method='set')
#
#
# test()
