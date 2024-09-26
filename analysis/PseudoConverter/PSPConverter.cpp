// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author DFTFE
//
#include <iostream>
#include <vector>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <stdexcept>
#include <cmath>

#include <utils/Exceptions.h>

std::vector<double>
XmlTagReaderMain(std::vector<std::string> tag_name,
                  std::string              file_path_in)
{
  xmlDocPtr  doc;
  xmlNodePtr cur;
  doc = xmlParseFile(file_path_in.c_str());
  cur = xmlDocGetRootElement(doc);
  // Finding the tag

  for (int i = 0; i < tag_name.size(); i++)
    {
      cur                 = cur->children;
      const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
      while (cur != NULL)
        {
          if ((!xmlStrcmp(cur->name, temp)))
            {
              break;
            }
          cur = cur->next;
        }
    }
  // If tag not found
  if (cur == NULL)
    throw std::invalid_argument("Tag not found");
  else
    {
      // Extracting main data
      xmlChar *key;
      key = xmlNodeListGetString(doc, cur->xmlChildrenNode, 1);
      std::string         main_str = (char *)key;
      std::vector<double> main;
      std::stringstream   ss;
      ss << main_str;
      double temp_str;
      while (!ss.eof())
        {
          ss >> temp_str;
          main.push_back(temp_str);
        }
      main.pop_back();
      return main;
    }
}

void
XmlTagReaderAttr(std::vector<std::string>  tag_name,
                  std::string               file_path_in,
                  std::vector<std::string> *attr_type,
                  std::vector<std::string> *attr_value)
{
  xmlDocPtr  doc;
  xmlNodePtr cur;
  doc = xmlParseFile(file_path_in.c_str());
  cur = xmlDocGetRootElement(doc);

  // Finding the tag
  for (int i = 0; i < tag_name.size(); i++)
    {
      cur                 = cur->children;
      const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
      while (cur != NULL)
        {
          if ((!xmlStrcmp(cur->name, temp)))
            {
              break;
            }
          cur = cur->next;
        }
    }

  // If tag not found
  if (cur == NULL)
    throw std::invalid_argument("Tag not found");
  else
    {
      // Extracting Attribute data
      xmlAttr *attribute = cur->properties;
      if (attribute == NULL)
        {
          throw std::invalid_argument("Tag does not have attributes");
        }
      else
        {
          for (xmlAttrPtr attr = cur->properties; NULL != attr;
                attr            = attr->next)
            {
              (*attr_type).push_back((char *)(attr->name));
              xmlChar *value = xmlNodeListGetString(doc, attr->children, 1);
              (*attr_value).push_back((char *)value);
            }
        }
    }
}
int
xmlNodeChildCount(std::vector<std::string> tag_name,
                  std::string              file_path_in)
{
  xmlDocPtr  doc;
  xmlNodePtr cur;
  doc = xmlParseFile(file_path_in.c_str());
  cur = xmlDocGetRootElement(doc);

  // Finding the tag
  for (int i = 0; i < tag_name.size(); i++)
    {
      cur                 = cur->children;
      const xmlChar *temp = (const xmlChar *)tag_name[i].c_str();
      while (cur != NULL)
        {
          if ((!xmlStrcmp(cur->name, temp)))
            {
              break;
            }
          cur = cur->next;
        }
    }
  // Counting children of current node
  int child_count = xmlChildElementCount(cur);
  return child_count;
}

void
xmltoLocalPotential(std::string file_path_in, std::string file_path_out)
{
  // Extracting radial coordinates
  std::vector<double>      radial_coord;
  std::vector<std::string> radial_tag;
  radial_tag.push_back("PP_MESH");
  radial_tag.push_back("PP_R");
  radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

  // Extracting local potential data
  std::vector<double>      local_pot_values;
  std::vector<std::string> local_por_tag;
  local_por_tag.push_back("PP_LOCAL");
  local_pot_values = XmlTagReaderMain(local_por_tag, file_path_in);

  // Writing the local potential data
  std::fstream file;
  file.open(file_path_out + "/locPot.dat", std::ios::out);
  file << std::setprecision(12);
  if (file.is_open())
    {
      for (int l = 0; l < radial_coord.size(); l++)
        {
          file << radial_coord[l] << " " << local_pot_values[l] / 2
                << std::endl;
        }
    }
  file.close();
}

void
xmltoDensityFile(std::string file_path_in, std::string file_path_out)
{
  // Extracting radial coordinates
  std::vector<double>      radial_coord;
  std::vector<std::string> radial_tag;
  radial_tag.push_back("PP_MESH");
  radial_tag.push_back("PP_R");
  radial_coord = XmlTagReaderMain(radial_tag, file_path_in);

  // Extracting valence density
  std::vector<double>      rhoatom_values;
  std::vector<std::string> rhoatom_tag;
  rhoatom_tag.push_back("PP_RHOATOM");
  rhoatom_values = XmlTagReaderMain(rhoatom_tag, file_path_in);

  // Writing density.inp
  double       pi = 2 * acos(0.0);
  std::fstream file;
  file.open(file_path_out + "/density.inp", std::ios::out);
  file << std::setprecision(15);
  if (file.is_open())
    {
      for (int l = 0; l < radial_coord.size(); l++)
        {
          if (l == 0)
            file << radial_coord[0] << " " << rhoatom_values[0]
                  << std::endl;
          else
            file << radial_coord[l] << " "
                  << rhoatom_values[l] /
                      (4 * pi * std::pow(radial_coord[l], 2))
                  << std::endl;
        }
    }
  file.close();
}
void
xmltoOrbitalFile(std::string file_path_in, std::string file_path_out)
{ // Extracting radial coordinates
  std::vector<double>      radial_coord;
  std::vector<std::string> radial_tag;
  radial_tag.push_back("PP_MESH");
  radial_tag.push_back("PP_R");
  radial_coord = XmlTagReaderMain(radial_tag, file_path_in);
  std::vector<std::string> pswfc_tag;
  pswfc_tag.push_back("PP_PSWFC");
  for (int i = 1; i <= xmlNodeChildCount(pswfc_tag, file_path_in); i++)
    {
      // Reading chi data
      std::string pp_chi_str = "PP_CHI.";
      pp_chi_str += std::to_string(i);
      std::vector<std::string> chi_tag;
      chi_tag.push_back("PP_PSWFC");
      chi_tag.push_back(pp_chi_str);
      std::vector<double> chi_values =
        XmlTagReaderMain(chi_tag, file_path_in);
      std::vector<std::string> attr_type;
      std::vector<std::string> attr_value;
      XmlTagReaderAttr(chi_tag, file_path_in, &attr_type, &attr_value);
      unsigned int index     = 0;
      std::string  to_search = "label";
      auto it = std::find(attr_type.begin(), attr_type.end(), to_search);
      if (it == attr_type.end())
        {
          throw std::invalid_argument("orbital label attribute not found");
        }
      else
        {
          index = std::distance(attr_type.begin(), it);
        }
      std::string orbital_string_nl = attr_value[index];
      for (auto &w : orbital_string_nl)
        {
          w = tolower(w);
        }
      char n = orbital_string_nl[0];
      char l;
      if (orbital_string_nl[1] == 's')
        {
          l = '0';
        }
      if (orbital_string_nl[1] == 'p')
        {
          l = '1';
        }
      if (orbital_string_nl[1] == 'd')
        {
          l = '2';
        }
      if (orbital_string_nl[1] == 'f')
        {
          l = '3';
        }
      std::string  orbital_string = "psi";
      std::fstream file;
      file.open(file_path_out + "/" + orbital_string + n + l + ".inp",
                std::ios::out);
      file << std::setprecision(12);
      if (file.is_open())
        {
          for (int l = 0; l < chi_values.size(); l++)
            {
              file << radial_coord[l] << " " << chi_values[l] << std::endl;
            }
        }
      file.close();
    }
}

void
pseudoPotentialToDftefeParser(const std::string file_path_in,
                              const std::string file_path_out)
{
  xmltoLocalPotential(file_path_in, file_path_out);
  xmltoDensityFile(file_path_in, file_path_out);
  xmltoOrbitalFile(file_path_in, file_path_out);
}

int main(int argc, char** argv)
{

  char* dftefe_path = getenv("DFTEFE_PATH");
  std::string sourceDir;
  // if executes if a non null value is returned
  // otherwise else executes
  if (dftefe_path != NULL) 
  {
    sourceDir = (std::string)dftefe_path + "/analysis/PseudoConverter/";
  }
  else
  {
    dftefe::utils::throwException(false,
                          "dftefe_path does not exist!");
  }
  std::string inDataFile = argv[1];
  std::string inputFileName = sourceDir + inDataFile;
  std::string outDataFile = argv[2];
  std::string outFileName = sourceDir + outDataFile;

  std::cout << "Reading input file: "<<inputFileName<<std::endl;
  std::cout << "Writing out file path: "<<outFileName<<std::endl;  
  
  pseudoPotentialToDftefeParser(inputFileName, outFileName);
  return 0;
}