/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Avirup Sircar
 */

#include <atoms/AtomSphericalDataPSP.h>
#include <utils/Exceptions.h>
#include <utils/StringOperations.h>
#include <sstream>
#include <vector>
#include <iterator>
#include <iomanip>
#include <cstring>
#include <string>

namespace dftefe
{
  namespace atoms
  {
    namespace AtomSphericalDataPSPXMLLocal
    {
      xmlDocPtr
      getDoc(const std::string fileName)
      {
        xmlDocPtr doc;
        doc = xmlParseFile(fileName.c_str());
        utils::throwException(doc != NULL,
                              "Error parsing the XML file:" + fileName);
        return doc;
      }

      xmlXPathObjectPtr
      getNodeSet(xmlDocPtr         doc,
                 const std::string xpath,
                 const std::string ns,
                 const std::string nsHRef)
      {
        xmlXPathContextPtr context;
        xmlXPathObjectPtr  result;
        context = xmlXPathNewContext(doc);
        utils::throwException(
          context != NULL,
          "Unable to create a new XPath context via xmlXPathNewContext()");

        xmlChar *nsXmlChar     = (xmlChar *)ns.c_str();
        xmlChar *nsHRefXmlChar = (xmlChar *)nsHRef.c_str();
        int      returnRegNS   = xmlXPathRegisterNs(context,
                                             BAD_CAST nsXmlChar,
                                             BAD_CAST nsHRefXmlChar);
        utils::throwException(returnRegNS == 0,
                              "Unable to register the with prefix " + ns + ":" +
                                nsHRef);

        xmlChar *xpathXmlChar = (xmlChar *)xpath.c_str();
        result                = xmlXPathEvalExpression(xpathXmlChar, context);
        xmlXPathFreeContext(context);
        utils::throwException(result != NULL,
                              "Unable to evaluate the XPath expression " +
                                xpath + " using xmlXPathEvalExpression");
        if (xmlXPathNodeSetIsEmpty(result->nodesetval))
          {
            xmlXPathFreeObject(result);
            utils::throwException(false,
                                  "XPath expression " + xpath + " not found.");
          }
        return result;
      }

      void
      getNodeStrings(const AtomSphericalDataPSP::XPathInfo &xPathInfo,
                     std::vector<std::string> &             nodeNames,
                     std::vector<std::string> &             nodeStrings)
      {
        nodeStrings.resize(0);
        nodeNames.resize(0);
        xmlNodeSetPtr     ptrToXmlNodeSet;
        xmlXPathObjectPtr ptrToXmlXPathObject;
        xmlChar *         keyword;
        ptrToXmlXPathObject = getNodeSet(xPathInfo.doc,
                                         xPathInfo.xpath,
                                         xPathInfo.ns,
                                         xPathInfo.nsHRef);
        ptrToXmlNodeSet     = ptrToXmlXPathObject->nodesetval;
        nodeStrings.resize(ptrToXmlNodeSet->nodeNr);
        nodeNames.resize(ptrToXmlNodeSet->nodeNr);
        for (size_type i = 0; i < ptrToXmlNodeSet->nodeNr; i++)
          {
            xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
            keyword                 = xmlNodeListGetString(xPathInfo.doc,
                                           ptrToXmlNode->xmlChildrenNode,
                                           1);
            keyword != NULL ? nodeStrings[i] = std::string((char *)keyword) :
                                              nodeStrings[i] = "";
            utils::stringOps::trim(nodeStrings[i]);
            xmlFree(keyword);
            const xmlChar *name = ptrToXmlNode->name;
            nodeNames[i]        = std::string((char *)name);
          }
        xmlXPathFreeObject(ptrToXmlXPathObject);
      }

      void
      getNodeStrings(
        const AtomSphericalDataPSP::XPathInfo &xPathInfo,
        std::vector<std::string> &             nodeNames,
        std::vector<std::string> &             nodeStrings,
        std::vector<std::vector<std::pair<std::string, std::string>>>
          &attrStrings)
      {
        nodeStrings.resize(0);
        attrStrings.resize(0);
        nodeNames.resize(0);
        xmlNodeSetPtr     ptrToXmlNodeSet;
        xmlXPathObjectPtr ptrToXmlXPathObject;
        xmlChar *         keyword;
        ptrToXmlXPathObject = getNodeSet(xPathInfo.doc,
                                         xPathInfo.xpath,
                                         xPathInfo.ns,
                                         xPathInfo.nsHRef);
        ptrToXmlNodeSet     = ptrToXmlXPathObject->nodesetval;
        nodeStrings.resize(ptrToXmlNodeSet->nodeNr);
        attrStrings.resize(ptrToXmlNodeSet->nodeNr);
        nodeNames.resize(ptrToXmlNodeSet->nodeNr);
        for (size_type i = 0; i < ptrToXmlNodeSet->nodeNr; i++)
          {
            xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
            keyword                 = xmlNodeListGetString(xPathInfo.doc,
                                           ptrToXmlNode->xmlChildrenNode,
                                           1);
            keyword != NULL ? nodeStrings[i] = std::string((char *)keyword) :
                                              nodeStrings[i] = "";
            utils::stringOps::trim(nodeStrings[i]);
            xmlFree(keyword);
            // loop over attributes
            xmlAttr *attribute = ptrToXmlNode->properties;
            std::vector<std::pair<std::string, std::string>> &attrPairs =
              attrStrings[i];
            while (attribute)
              {
                const xmlChar *attrName = attribute->name;
                xmlChar *   attrValue = xmlNodeListGetString(ptrToXmlNode->doc,
                                                          attribute->children,
                                                          1);
                std::string name((const char *)attrName);
                std::string value(attrValue != NULL ? (char *)attrValue : "");
                utils::stringOps::trim(value);
                attrPairs.push_back(std::make_pair(name, value));
                xmlFree(attrValue);
                attribute = attribute->next;
              }
            const xmlChar *name = ptrToXmlNode->name;
            nodeNames[i]        = std::string((char *)name);
          }
        xmlXPathFreeObject(ptrToXmlXPathObject);
      }

      void
      getChildrenNodeStrings(
        const AtomSphericalDataPSP::XPathInfo &xPathInfo,
        std::vector<std::string> &             nodeNames,
        std::vector<std::string> &             nodeStrings,
        std::vector<std::vector<std::pair<std::string, std::string>>>
          &attrStrings)
      {
        nodeStrings.resize(0);
        attrStrings.resize(0);
        nodeNames.resize(0);
        xmlNodeSetPtr     ptrToXmlNodeSet;
        xmlXPathObjectPtr ptrToXmlXPathObject;
        xmlChar *         keyword;
        ptrToXmlXPathObject = getNodeSet(xPathInfo.doc,
                                         xPathInfo.xpath,
                                         xPathInfo.ns,
                                         xPathInfo.nsHRef);
        ptrToXmlNodeSet     = ptrToXmlXPathObject->nodesetval;
        utils::throwException(
          ptrToXmlNodeSet->nodeNr == 1,
          "Found more than one " + xPathInfo.xpath + " element in " +
            xPathInfo.fileName +
            " while getting children nodes. Extend the function to handle this. ");
        for (size_type i = 0; i < ptrToXmlNodeSet->nodeNr; i++)
          {
            int countChildren = 0;
            for (xmlNodePtr ptrToXmlNode =
                   ptrToXmlNodeSet->nodeTab[i]->children;
                 ptrToXmlNode;
                 ptrToXmlNode = ptrToXmlNode->next)
              {
                if (ptrToXmlNode->type == XML_ELEMENT_NODE)
                  {
                    countChildren += 1;
                  }
              }
            nodeStrings.resize(countChildren);
            attrStrings.resize(countChildren);
            nodeNames.resize(countChildren);
            int count = 0;
            for (xmlNodePtr ptrToXmlNode =
                   ptrToXmlNodeSet->nodeTab[i]->children;
                 ptrToXmlNode;
                 ptrToXmlNode = ptrToXmlNode->next)
              {
                if (ptrToXmlNode->type == XML_ELEMENT_NODE)
                  {
                    keyword =
                      xmlNodeListGetString(xPathInfo.doc,
                                           ptrToXmlNode->xmlChildrenNode,
                                           1);
                    keyword != NULL ?
                      nodeStrings[count] = std::string((char *)keyword) :
                      nodeStrings[count] = "";
                    utils::stringOps::trim(nodeStrings[count]);
                    xmlFree(keyword);
                    // loop over attributes
                    xmlAttr *attribute = ptrToXmlNode->properties;
                    std::vector<std::pair<std::string, std::string>>
                      &attrPairs = attrStrings[count];
                    while (attribute)
                      {
                        const xmlChar *attrName = attribute->name;
                        xmlChar *      attrValue =
                          xmlNodeListGetString(ptrToXmlNode->doc,
                                               attribute->children,
                                               1);
                        std::string name((const char *)attrName);
                        std::string value(
                          attrValue != NULL ? (char *)attrValue : "");
                        utils::stringOps::trim(value);
                        attrPairs.push_back(std::make_pair(name, value));
                        xmlFree(attrValue);
                        attribute = attribute->next;
                      }
                    const xmlChar *name = ptrToXmlNode->name;
                    nodeNames[count]    = std::string((char *)name);
                    count += 1;
                  }
              }
          }
        xmlXPathFreeObject(ptrToXmlXPathObject);
      }
    } // namespace AtomSphericalDataPSPXMLLocal

    namespace
    {
      std::string
      getXPath(const std::string &rootElementName,
               const std::string &elementName)
      {
        auto it =
          AtomSphDataPSPDefaults::UPF_PARENT_PATH_LOOKUP.find(elementName);
        utils::throwException(
          it != AtomSphDataPSPDefaults::UPF_PARENT_PATH_LOOKUP.end(),
          "No element name '" + elementName +
            "' found in UPF_PARENT_PATH_LOOKUP. The fieldNames vector for UPF are standarized by authors.");
        return rootElementName + "/" + it->second;
      }

      void
      storeQNumbersToDataIdMap(
        const std::vector<std::shared_ptr<SphericalData>> &sphericalDataVec,
        std::map<std::vector<int>, size_type> &            qNumbersToDataIdMap)
      {
        size_type N = sphericalDataVec.size();
        for (size_type i = 0; i < N; ++i)
          {
            qNumbersToDataIdMap[sphericalDataVec[i]->getQNumbers()] = i;
          }
      }
    } // namespace

    AtomSphericalDataPSP::AtomSphericalDataPSP(
      const std::string                 fileName,
      const std::vector<std::string> &  fieldNames,
      const std::vector<std::string> &  metadataNames,
      const SphericalHarmonicFunctions &sphericalHarmonicFunc)
      : d_fileName(fileName)
      , d_fieldNames(fieldNames)
      , d_metadataNames(metadataNames)
      , d_scalarSpatialFnAfterRadialGrid(nullptr)
      , d_sphericalHarmonicFunc(sphericalHarmonicFunc)
      , d_PSPVLocalCutoff(10.0001) // bohr
    {
#if defined(LIBXML_XPATH_ENABLED) && defined(LIBXML_SAX1_ENABLED)
      xmlDocPtr ptrToXmlDoc;
      d_rootElementName  = "/UPF";
      std::string ns     = "dummy";
      std::string nsHRef = "dummy";
      ptrToXmlDoc        = AtomSphericalDataPSPXMLLocal::getDoc(fileName);

      XPathInfo xPathInfo;
      xPathInfo.fileName = fileName;
      xPathInfo.doc      = ptrToXmlDoc;
      xPathInfo.ns       = ns;
      xPathInfo.nsHRef   = nsHRef;

      std::vector<std::string> nodeStrings(0);
      std::vector<std::string> nodeNames(0);
      std::vector<std::vector<std::pair<std::string, std::string>>> attrStrings(
        0);

      bool convSuccess = false;

      int numRadialPoints = 0;
      d_zvalance          = 0.;
      d_lmax              = 0.;
      d_numProj           = 0;

      //
      // get metadata in PP_HEADER
      //
      xPathInfo.xpath = d_rootElementName + "/PP_HEADER";
      AtomSphericalDataPSPXMLLocal::getNodeStrings(xPathInfo,
                                                   nodeNames,
                                                   nodeStrings,
                                                   attrStrings);
      utils::throwException(attrStrings.size() == 1,
                            "Found more than one " + xPathInfo.xpath +
                              " element in " + fileName);

      for (int i = 0; i < attrStrings[0].size(); i++)
        {
          if (attrStrings[0][i].first == "mesh_size")
            {
              convSuccess = utils::stringOps::strToInt(attrStrings[0][i].second,
                                                       numRadialPoints);
              utils::throwException(convSuccess,
                                    "Error while converting " +
                                      xPathInfo.xpath + " element in " +
                                      fileName + " to int");
              utils::throwException(numRadialPoints > 0,
                                    "Non-positive numRadial points found for" +
                                      xPathInfo.xpath + " element in " +
                                      fileName);
            }
          if (attrStrings[0][i].first == "z_valence")
            {
              convSuccess =
                utils::stringOps::strToDouble(attrStrings[0][i].second,
                                              d_zvalance);
              utils::throwException(convSuccess,
                                    "Error while converting " +
                                      xPathInfo.xpath + " element in " +
                                      fileName + " to int");
              utils::throwException(d_zvalance > 0,
                                    "Non-positive z valance found for" +
                                      xPathInfo.xpath + " element in " +
                                      fileName);
            }
          if (attrStrings[0][i].first == "l_max")
            {
              convSuccess =
                utils::stringOps::strToInt(attrStrings[0][i].second, d_lmax);
              utils::throwException(convSuccess,
                                    "Error while converting " +
                                      xPathInfo.xpath + " element in " +
                                      fileName + " to int");
              utils::throwException(d_lmax >= 0,
                                    "Non-positive lmax found for" +
                                      xPathInfo.xpath + " element in " +
                                      fileName);
            }
            if (attrStrings[0][i].first == "number_of_proj")
            {
              convSuccess =
                utils::stringOps::strToInt(attrStrings[0][i].second, d_numProj);
              utils::throwException(convSuccess,
                                    "Error while converting " +
                                      xPathInfo.xpath + " element in " +
                                      fileName + " to int");
              utils::throwException(d_numProj >= 0,
                                    "Non-positive lmax found for" +
                                      xPathInfo.xpath + " element in " +
                                      fileName);
            }
        }
      for (int j = 0; j < d_metadataNames.size(); j++)
        {
          bool foundMetaData = false;
          for (int i = 0; i < attrStrings[0].size(); i++)
            {
              if (attrStrings[0][i].first == d_metadataNames[j])
                {
                  d_metadata[d_metadataNames[j]] = attrStrings[0][i].second;
                  foundMetaData                  = true;
                  break;
                }
              else if ("dij" == d_metadataNames[j])
                {
                  d_metadata["dij"] = "";
                  foundMetaData     = true;
                  break;
                }
              else
                continue;
            }
          utils::throwException(
            foundMetaData == true,
            "No metadata name '" + d_metadataNames[j] +
              "' found matching with the given metadata vector. The metadatanames vector for UPF are standarized by authors.");
        }

      std::vector<double> radialPoints;
      // get radial points
      //
      xPathInfo.xpath = getXPath(d_rootElementName, "r") + "/PP_R";
      AtomSphericalDataPSPXMLLocal::getNodeStrings(xPathInfo,
                                                   nodeNames,
                                                   nodeStrings);
      utils::throwException(nodeStrings.size() == 1,
                            "Found more than one " + xPathInfo.xpath +
                              " element in " + fileName);
      radialPoints.resize(0);
      convSuccess = utils::stringOps::splitStringToDoubles(nodeStrings[0],
                                                           radialPoints,
                                                           numRadialPoints);
      utils::throwException(convSuccess,
                            "Error while converting " + xPathInfo.xpath +
                              " element in " + fileName + " to double");
      utils::throwException(
        radialPoints.size() == numRadialPoints,
        "Mismatch in number of radial points specified and the number of "
        " radial points provided in " +
          fileName);

      //
      // store field spherical data
      //
      for (size_type iField = 0; iField < fieldNames.size(); ++iField)
        {
          const std::string fieldName = fieldNames[iField];
          std::vector<std::shared_ptr<SphericalData>> sphericalDataVec(0);
          std::map<std::vector<int>, size_type>       qNumbersToIdMap;
          getSphericalDataFromXMLNode(sphericalDataVec,
                                      radialPoints,
                                      xPathInfo,
                                      fieldName,
                                      sphericalHarmonicFunc);
          storeQNumbersToDataIdMap(sphericalDataVec, qNumbersToIdMap);
          d_sphericalData[fieldName]   = sphericalDataVec;
          d_qNumbersToIdMap[fieldName] = qNumbersToIdMap;
        }

      xmlFreeDoc(ptrToXmlDoc);
      xmlCleanupParser();

      d_radialPoints = radialPoints;
#else
      utils::throwException(
        false, "Support for LIBXML XPATH and LIBXML SAX1 not found");
#endif // defined(LIBXML_XPATH_ENABLED) && defined(LIBXML_SAX1_ENABLED)
    }

    void
    AtomSphericalDataPSP::getSphericalDataFromXMLNode(
      std::vector<std::shared_ptr<SphericalData>> &sphericalDataVec,
      const std::vector<double> &                  radialPoints,
      XPathInfo &                                  xPathInfo,
      const std::string &                          fieldName,
      const SphericalHarmonicFunctions &           sphericalHarmonicFunc)
    {
      const size_type numPoints = radialPoints.size();
      size_type       N;

      xPathInfo.xpath = getXPath(d_rootElementName, fieldName);

      std::vector<std::string> nodeStrings(0);
      std::vector<std::string> nodeNames(0);
      std::vector<std::vector<std::pair<std::string, std::string>>> attrStrings(
        0); // numNodes x numAttrInEachNode x attrPair

      sphericalDataVec.resize(0);

      std::vector<std::vector<int>> qNumbersVec(0);
      bool                          convSuccess = false;

      bool isCoreCorrectAttributePresent = false;
      utils::stringOps::strToBool(d_metadata["core_correction"],
        isCoreCorrectAttributePresent);

      // get quantum Numbers vec
      std::vector<int> nodeIndex(0);
      if (fieldName == std::string("beta"))
        {
          AtomSphericalDataPSPXMLLocal::getChildrenNodeStrings(xPathInfo,
                                                               nodeNames,
                                                               nodeStrings,
                                                               attrStrings);
          N     = nodeStrings.size();
          int l = 0, p = 0, lPrev = 0;
          int maxlFromBeta = 0;
          for (size_type i = 0; i < N; ++i)
            {
              if (nodeNames[i].find("PP_BETA") != std::string::npos)
                {
                  for (auto &attrId : attrStrings[i])
                    {
                      if (attrId.first == "angular_momentum")
                        {
                          utils::stringOps::strToInt(attrId.second, l);
                          utils::throwException(l >= 0,
                                                "Non-positive l found for" +
                                                  xPathInfo.xpath +
                                                  " element in " +
                                                  xPathInfo.fileName);
                          break;
                        }
                    }

                  if (i != 0)
                    {
                      p     = (l != lPrev) ? p = 0 : p += 1;
                      lPrev = l;
                    }

                  for (int m = -l; m <= l; m++)
                    {
                      nodeIndex.push_back(i);
                      std::vector<int> qNumbers = {p, l, m};
                      utils::throwException(
                        find(qNumbersVec.begin(),
                             qNumbersVec.end(),
                             qNumbers) == qNumbersVec.end(),
                        "Trying to enter more than one " + xPathInfo.xpath +
                          " of same quantum number vector from " +
                          xPathInfo.fileName);
                      qNumbersVec.push_back(qNumbers);
                    }
                  if (maxlFromBeta < l)
                    maxlFromBeta = l;
                }
              else if (nodeNames[i].find("PP_DIJ") != std::string::npos)
                {             
                  std::vector<double> dijs(0);
                  convSuccess =
                    utils::stringOps::splitStringToDoubles(nodeStrings[i],
                                                           dijs,
                                                           d_numProj*d_numProj);
                  utils::throwException(convSuccess,
                                        "Error while converting values in " +
                                          xPathInfo.xpath + " element in " +
                                          xPathInfo.fileName + " to double");          
                  for(auto &i : dijs)
                  {
                    i *= 0.5;  // convert from Ry to Ha
                  }   
                  std::stringstream ss;
                  ss << std::fixed << std::setprecision(16);
                  std::transform(dijs.begin(), dijs.end(), std::ostream_iterator<double>(ss, " "), [](double d){return d;});
                  d_metadata["dij"] = ss.str();

                  //d_metadata["dij"] = nodeStrings[i];
                }
            }
          utils::throwException(maxlFromBeta == d_lmax,
                                "The lmax from PSP HEADER does not match from" +
                                  xPathInfo.xpath + " element in " +
                                  xPathInfo.fileName);
        }
      else if (fieldName == std::string("pswfc"))
        {
          AtomSphericalDataPSPXMLLocal::getChildrenNodeStrings(xPathInfo,
                                                               nodeNames,
                                                               nodeStrings,
                                                               attrStrings);
          N = nodeStrings.size();
          int n, l;
          for (size_type i = 0; i < N; ++i)
            {
              if (nodeNames[i].find("PP_CHI") != std::string::npos)
                {
                  for (auto &attrId : attrStrings[i])
                    {
                      if (attrId.first == "l")
                        {
                          utils::stringOps::strToInt(attrId.second, l);
                          break;
                        }
                    }
                  for (auto attrId : attrStrings[i])
                    {
                      if (attrId.first == "label")
                        {
                          utils::stringOps::strToInt(
                            std::string(1, attrId.second[0]), n);
                          break;
                        }
                    }
                  for (int m = -l; m <= l; m++)
                    {
                      nodeIndex.push_back(i);
                      std::vector<int> qNumbers = {n, l, m};
                      utils::throwException(
                        find(qNumbersVec.begin(),
                             qNumbersVec.end(),
                             qNumbers) == qNumbersVec.end(),
                        "Trying to enter more than one " + xPathInfo.xpath +
                          " of same quantum number vector from " +
                          xPathInfo.fileName);
                      qNumbersVec.push_back(qNumbers);
                    }
                }
            }
        }
      else if(fieldName == std::string("nlcc") && !isCoreCorrectAttributePresent)
        {
          // since some PSP files may have core_correction and some may not in a KSDFT calculation
          std::string stringsZero = [numPoints]{ std::string s; for (int i = 0; i < numPoints; ++i) s += (i ? " 0" : "0"); return s; }();
          nodeStrings.resize(1);
          nodeStrings[0] = stringsZero;
          nodeIndex.push_back(0);
          std::vector<int> qNumbers = {0, 0, 0};
          utils::throwException(
            find(qNumbersVec.begin(), qNumbersVec.end(), qNumbers) ==
              qNumbersVec.end(),
            "Trying to enter more than one " + xPathInfo.xpath +
              " of same quantum number vector from " + xPathInfo.fileName);
          qNumbersVec.push_back(qNumbers);
        }
      else
        {
          AtomSphericalDataPSPXMLLocal::getNodeStrings(xPathInfo,
                                                       nodeNames,
                                                       nodeStrings);
          N = nodeStrings.size();
          utils::throwException(N == 1,
                                "Found more than one " + xPathInfo.xpath +
                                  " element in " + xPathInfo.fileName);
          nodeIndex.push_back(0);
          std::vector<int> qNumbers = {0, 0, 0};
          utils::throwException(
            find(qNumbersVec.begin(), qNumbersVec.end(), qNumbers) ==
              qNumbersVec.end(),
            "Trying to enter more than one " + xPathInfo.xpath +
              " of same quantum number vector from " + xPathInfo.fileName);
          qNumbersVec.push_back(qNumbers);
        }

      N = qNumbersVec.size();
      for (size_type i = 0; i < N; ++i)
        {
          std::vector<double> radialValues(0);
          convSuccess =
            utils::stringOps::splitStringToDoubles(nodeStrings[nodeIndex[i]],
                                                   radialValues,
                                                   numPoints);
          utils::throwException(convSuccess,
                                "Error while converting values in " +
                                  xPathInfo.xpath + " element in " +
                                  xPathInfo.fileName + " to double");
          utils::throwException(
            radialValues.size() == numPoints,
            "Mismatch in number of points specified and number of points "
            "provided in " +
              xPathInfo.xpath + " element in " + xPathInfo.fileName);

          // transform the radialValues by the following constants based on UPF
          // format
          if (fieldName == std::string("vlocal")) // multiply by 0.5
            {
              double constant = 0.5;
              std::transform(radialValues.begin(),
                             radialValues.end(),
                             radialValues.begin(),
                             [constant](double element) {
                               return element * constant;
                             });
            }
          else if ((fieldName == std::string("beta")) ||
                   (fieldName ==
                     std::string("pswfc"))) // divide by rGrid ; rGridVal(0) = rGridVal(1)
            {
              for (int i = 1; i < radialValues.size(); i++)
                {
                  radialValues[i] = radialValues[i] / radialPoints[i];
                }
              radialValues[0] = radialValues[1];
            }
          else if (fieldName ==
                   std::string("rhoatom")) // divide by 4pir^2 ; rGridVal(0) = rGridVal(1)
            {
              for (int i = 1; i < radialValues.size(); i++)
                {
                  radialValues[i] =
                    radialValues[i] / (4 * utils::mathConstants::pi *
                                       radialPoints[i] * radialPoints[i]);
                }
              radialValues[0] = radialValues[1];
            }

          // get cutoff from the radial values
          // if(val < 1e-10 cutoff else cutoff = 1e10)
          double cutoff = 1.0e10;
          for (int j = radialValues.size() - 1; j > 0; j--)
            if (std::abs(radialValues[j]) >= 1e-10)
              {
                (j != radialValues.size() - 1) ? cutoff = radialPoints[j] :
                                                 cutoff = 1.0e10;
                break;
              }

          if (fieldName == std::string("rhoatom"))
              cutoff = radialPoints.back();

          // std::cout << fieldName << " : " << qNumbersVec[i][0] << ","
          //           << qNumbersVec[i][1] << "," << qNumbersVec[i][2] << " ; "
          //           << cutoff << "\n";

          // multiply by 1/y_00 to get rid of the constant in f(r)*Y_lm
          double constant =
            1. / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0));
          //if (fieldName == "vlocal" || fieldName == "nlcc" || fieldName == "nlcc")
          if ((fieldName != std::string("beta")) && (fieldName != std::string("pswfc")))
          {
            std::transform(radialValues.begin(),
                           radialValues.end(),
                           radialValues.begin(),
                           [constant](double &element) {
                             return element * constant;
                           });
          }
          // NOTE : for vlocal the function is mixed, modify accordingly

          // if(fieldName == "beta")
          // std::cout << radialPoints[0] << "\t" << radialValues[0] << "\n";

          double smoothness = 1.e10;
          // create spherical data vec
          if (fieldName == std::string("vlocal"))
            {
              std::vector<double> radialPointsWithCutoff(0);
              std::vector<double> radialValuesWithCutoff(0);

              // Larger max allowed Tail is important for pseudo-dojo database
              // ONCV pseudopotential local potentials which have a larger data
              // range with slow convergence to -Z/r Same value of 10.0 used as
              // rcut in QUANTUM ESPRESSO (cf. Modules/read_pseudo.f90)

              for (int i = 0; i < radialValues.size(); i++)
                {
                  if (radialPoints[i] <= d_PSPVLocalCutoff)
                    {
                      radialPointsWithCutoff.push_back(radialPoints[i]);
                      radialValuesWithCutoff.push_back(radialValues[i]);
                    }
                }

              d_scalarSpatialFnAfterRadialGrid =
                std::make_shared<utils::PointChargePotentialFunction>(
                  utils::Point({0, 0, 0}),
                  -1.0 * constant * std::abs(d_zvalance));


              double leftValue =
                (radialValuesWithCutoff[1] - radialValuesWithCutoff[0]) /
                (radialPointsWithCutoff[1] - radialPointsWithCutoff[0]);
              double rightValue =
                (-1.0) * std::abs(radialValuesWithCutoff.back() /
                                  radialPointsWithCutoff.back());

              utils::Spline::bd_type left = utils::Spline::bd_type::first_deriv;
              utils::Spline::bd_type right =
                utils::Spline::bd_type::first_deriv;

              sphericalDataVec.push_back(std::make_shared<SphericalDataMixed>(
                qNumbersVec[i],
                radialPointsWithCutoff,
                radialValuesWithCutoff,
                left,
                leftValue,
                right,
                rightValue,
                *d_scalarSpatialFnAfterRadialGrid,
                sphericalHarmonicFunc));
            }
          else
            {
              sphericalDataVec.push_back(
                std::make_shared<SphericalDataNumerical>(
                  qNumbersVec[i],
                  radialPoints,
                  radialValues,
                  cutoff,
                  smoothness,
                  sphericalHarmonicFunc));
            }
        }
    }

    void
    AtomSphericalDataPSP::addFieldName(const std::string fieldName)
    {
      if (std::find(d_fieldNames.begin(), d_fieldNames.end(), fieldName) !=
          d_fieldNames.end())
        {
          utils::throwException(false,
                                "Field " + fieldName +
                                  " is already given to AtomSphericalDataPSP " +
                                  d_fileName);
        }
      else
        {
          xmlDocPtr ptrToXmlDoc;
          d_rootElementName  = "/UPF";
          std::string ns     = "dummy";
          std::string nsHRef = "dummy";
          ptrToXmlDoc        = AtomSphericalDataPSPXMLLocal::getDoc(d_fileName);

          XPathInfo xPathInfo;
          xPathInfo.fileName = d_fileName;
          xPathInfo.doc      = ptrToXmlDoc;
          xPathInfo.ns       = ns;
          xPathInfo.nsHRef   = nsHRef;

          std::vector<std::shared_ptr<SphericalData>> sphericalDataVec(0);
          std::map<std::vector<int>, size_type>       qNumbersToIdMap;
          getSphericalDataFromXMLNode(sphericalDataVec,
                                      d_radialPoints,
                                      xPathInfo,
                                      fieldName,
                                      d_sphericalHarmonicFunc);
          storeQNumbersToDataIdMap(sphericalDataVec, qNumbersToIdMap);
          d_sphericalData[fieldName]   = sphericalDataVec;
          d_qNumbersToIdMap[fieldName] = qNumbersToIdMap;

          xmlFreeDoc(ptrToXmlDoc);
          xmlCleanupParser();
        }
    }

    std::string
    AtomSphericalDataPSP::getFileName() const
    {
      return d_fileName;
    }

    std::vector<std::string>
    AtomSphericalDataPSP::getFieldNames() const
    {
      return d_fieldNames;
    }

    std::vector<std::string>
    AtomSphericalDataPSP::getMetadataNames() const
    {
      return d_metadataNames;
    }

    const std::vector<std::shared_ptr<SphericalData>> &
    AtomSphericalDataPSP::getSphericalData(const std::string fieldName) const
    {
      auto it = d_sphericalData.find(fieldName);
      DFTEFE_AssertWithMsg(it != d_sphericalData.end(),
                           ("FieldName " + fieldName +
                            " not while parsing the XML file:" + d_fileName)
                             .c_str());
      return it->second;
    }


    const std::shared_ptr<SphericalData>
    AtomSphericalDataPSP::getSphericalData(
      const std::string       fieldName,
      const std::vector<int> &qNumbers) const
    {
      auto it = d_sphericalData.find(fieldName);
      DFTEFE_AssertWithMsg(it != d_sphericalData.end(),
                           ("Unable to find the field " + fieldName +
                            " while parsing the XML file " + d_fileName)
                             .c_str());
      auto iter = d_qNumbersToIdMap.find(fieldName);
      DFTEFE_AssertWithMsg(iter != d_qNumbersToIdMap.end(),
                           ("Unable to find the field " + fieldName +
                            " while parsing the XML file " + d_fileName)
                             .c_str());
      auto iterQNumberToId = (iter->second).find(qNumbers);
      if (iterQNumberToId != (iter->second).end())
        return *((it->second).begin() + iterQNumberToId->second);
      else
        {
          std::string s = "";
          for (size_type i = 0; i < qNumbers.size(); i++)
            s += std::to_string(qNumbers[i]) + " ";

          DFTEFE_AssertWithMsg(false,
                               ("Unable to find the qNumbers " + s + " for " +
                                " the field " + fieldName +
                                " while parsing the XML file " + d_fileName)
                                 .c_str());
          return *((it->second).begin() + iterQNumberToId->second);
        }
    }

    std::string
    AtomSphericalDataPSP::getMetadata(const std::string metadataName) const
    {
      auto it = d_metadata.find(metadataName);
      utils::throwException(it != d_metadata.end(),
                            "Unable to find the metadata " + metadataName +
                              " while parsing the XML file " + d_fileName);

      return it->second;
    }


    size_type
    AtomSphericalDataPSP::getQNumberID(const std::string       fieldName,
                                       const std::vector<int> &qNumbers) const
    {
      auto it = d_qNumbersToIdMap.find(fieldName);
      utils::throwException<utils::InvalidArgument>(
        it != d_qNumbersToIdMap.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataPSP::getQNumberID");
      auto it1 = (it->second).find(qNumbers);
      utils::throwException<utils::InvalidArgument>(
        it1 != (it->second).end(),
        "Cannot find the qnumbers provided to AtomSphericalDataPSP::getQNumberID");
      return (it1)->second;
    }

    size_type
    AtomSphericalDataPSP::nSphericalData(std::string fieldName) const
    {
      auto it = d_sphericalData.find(fieldName);
      utils::throwException<utils::InvalidArgument>(
        it != d_sphericalData.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataPSP::nSphericalData");
      return (it->second).size();
    }

  } // end of namespace atoms
} // end of namespace dftefe
