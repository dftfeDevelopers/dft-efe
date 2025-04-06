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
 * @author Bikash Kanungo
 */

#include <atoms/AtomSphericalData.h>
#include <utils/Exceptions.h>
#include <utils/StringOperations.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/tree.h>
#include <libxml/xpathInternals.h>
#include <sstream>
#include <set>
namespace dftefe
{
  namespace atoms
  {
    namespace AtomSphericalDataXMLLocal
    {
      struct XPathInfo
      {
        xmlDocPtr   doc;
        std::string fileName;
        std::string xpath;
        std::string ns;
        std::string nsHRef;
      };

      // bool
      // splitStringToInts(const std::string s,
      //                   std::vector<int> &vals,
      //                   size_type         reserveSize = 0)
      // {
      //   std::istringstream ss(s);
      //   std::string        word;
      //   size_type          wordCount = 0;
      //   vals.resize(0);
      //   vals.reserve(reserveSize);
      //   bool convSuccess = false;
      //   int  x;
      //   while (ss >> word)
      //     {
      //       convSuccess = utils::stringOps::strToInt(word, x);
      //       if (!convSuccess)
      //         break;
      //       vals.push_back(x);
      //     }
      //   return convSuccess;
      // }

      // bool
      // splitStringToDoubles(const std::string    s,
      //                      std::vector<double> &vals,
      //                      size_type            reserveSize = 0)
      // {
      //   std::istringstream ss(s);
      //   std::string        word;
      //   size_type          wordCount = 0;
      //   vals.resize(0);
      //   vals.reserve(reserveSize);
      //   bool   convSuccess = false;
      //   double x;
      //   while (ss >> word)
      //     {
      //       convSuccess = utils::stringOps::strToDouble(word, x);
      //       if (!convSuccess)
      //         break;
      //       vals.push_back(x);
      //     }
      //   return convSuccess;
      // }

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
      getNodeStrings(const XPathInfo &         xPathInfo,
                     std::vector<std::string> &nodeStrings)
      {
        nodeStrings.resize(0);
        xmlNodeSetPtr     ptrToXmlNodeSet;
        xmlXPathObjectPtr ptrToXmlXPathObject;
        xmlChar *         keyword;
        ptrToXmlXPathObject = getNodeSet(xPathInfo.doc,
                                         xPathInfo.xpath,
                                         xPathInfo.ns,
                                         xPathInfo.nsHRef);
        ptrToXmlNodeSet     = ptrToXmlXPathObject->nodesetval;
        nodeStrings.resize(ptrToXmlNodeSet->nodeNr);
        for (size_type i = 0; i < ptrToXmlNodeSet->nodeNr; i++)
          {
            keyword =
              xmlNodeListGetString(xPathInfo.doc,
                                   ptrToXmlNodeSet->nodeTab[i]->xmlChildrenNode,
                                   1);
            nodeStrings[i] = std::string((char *)keyword);
            xmlFree(keyword);
          }
        xmlXPathFreeObject(ptrToXmlXPathObject);
      }

      void
      getNodeStrings(
        const XPathInfo &         xPathInfo,
        std::vector<std::string> &nodeStrings,
        std::vector<std::vector<std::pair<std::string, std::string>>>
          &attrStrings)
      {
        nodeStrings.resize(0);
        attrStrings.resize(0);
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
        for (size_type i = 0; i < ptrToXmlNodeSet->nodeNr; i++)
          {
            xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
            keyword                 = xmlNodeListGetString(xPathInfo.doc,
                                           ptrToXmlNode->xmlChildrenNode,
                                           1);
            nodeStrings[i]          = std::string((char *)keyword);
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
                std::string value((char *)attrValue);
                attrPairs.push_back(std::make_pair(name, value));
                xmlFree(attrValue);
                attribute = attribute->next;
              }
          }
        xmlXPathFreeObject(ptrToXmlXPathObject);
      }

      void
      readSphericalDataFromXMLNodeData(
        std::vector<std::string> &radialValuesStrings,
        std::vector<std::string> &qNumbersStrings,
        std::vector<std::string> &cutOffInfoStrings,
        const XPathInfo &         xPathInfo)
      {
        radialValuesStrings.resize(0);
        qNumbersStrings.resize(0);
        cutOffInfoStrings.resize(0);
        xmlNodeSetPtr     ptrToXmlNodeSet;
        xmlXPathObjectPtr ptrToXmlXPathObject;
        xmlChar *         keyword;
        std::string xpathPrefix = xPathInfo.xpath + "/" + xPathInfo.ns + ":";
        int         numNodes    = 0;

        // get the values
        std::string valuesTag = "values";
        ptrToXmlXPathObject   = getNodeSet(xPathInfo.doc,
                                         xpathPrefix + valuesTag,
                                         xPathInfo.ns,
                                         xPathInfo.nsHRef);
        ptrToXmlNodeSet       = ptrToXmlXPathObject->nodesetval;
        numNodes              = ptrToXmlNodeSet->nodeNr;
        radialValuesStrings.resize(numNodes);
        for (size_type i = 0; i < numNodes; i++)
          {
            xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
            keyword                 = xmlNodeListGetString(xPathInfo.doc,
                                           ptrToXmlNode->xmlChildrenNode,
                                           1);
            radialValuesStrings[i]  = std::string((char *)keyword);
            xmlFree(keyword);
          }
        xmlXPathFreeObject(ptrToXmlXPathObject);

        // get the qNumbers
        std::string qNumbersTag = "qNumbers";
        ptrToXmlXPathObject     = getNodeSet(xPathInfo.doc,
                                         xpathPrefix + qNumbersTag,
                                         xPathInfo.ns,
                                         xPathInfo.nsHRef);
        ptrToXmlNodeSet         = ptrToXmlXPathObject->nodesetval;
        numNodes                = ptrToXmlNodeSet->nodeNr;
        qNumbersStrings.resize(numNodes);
        for (size_type i = 0; i < numNodes; i++)
          {
            xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
            keyword                 = xmlNodeListGetString(xPathInfo.doc,
                                           ptrToXmlNode->xmlChildrenNode,
                                           1);
            qNumbersStrings[i]      = std::string((char *)keyword);
            xmlFree(keyword);
          }
        xmlXPathFreeObject(ptrToXmlXPathObject);

        // get the cutoff infor
        std::string cutOffInfoTag = "cutoffInfo";
        ptrToXmlXPathObject       = getNodeSet(xPathInfo.doc,
                                         xpathPrefix + cutOffInfoTag,
                                         xPathInfo.ns,
                                         xPathInfo.nsHRef);
        ptrToXmlNodeSet           = ptrToXmlXPathObject->nodesetval;
        numNodes                  = ptrToXmlNodeSet->nodeNr;
        cutOffInfoStrings.resize(numNodes);
        for (size_type i = 0; i < numNodes; i++)
          {
            xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
            keyword                 = xmlNodeListGetString(xPathInfo.doc,
                                           ptrToXmlNode->xmlChildrenNode,
                                           1);
            cutOffInfoStrings[i]    = std::string((char *)keyword);
            xmlFree(keyword);
          }
        xmlXPathFreeObject(ptrToXmlXPathObject);
      }

      //      void
      //	processSphericalDataFromXMLNodeData(std::map<std::vector<int>,
      // std::vector<double>> & d_qNumbersToRadialField, 	    const
      // std::vector<std::string> & nodeStrings, 	    const
      // std::vector<std::vector<std::pair<std::string, std::string>>> &
      // attrStrings, 	    const XPathInfo & xPathInfo, 	    const size_type
      // numPoints)
      //	{
      //
      //	  const size_type N  = nodeStrings.size();
      //	  std::set<std::vector<int>> qNumbersSet;
      //	  bool convSuccess = false;
      //	  for(size_type i = 0; i < N; ++i)
      //	  {
      //	    std::vector<double> vals(0);
      //	    convSuccess =
      // utils::stringOps::splitStringToDoubles(nodeStrings[0], vals,
      // numPoints);
      //      	    utils::throwException(convSuccess,
      //	  	"Error while converting values in " + xPathInfo.xpath + " element
      // in " + xPathInfo.fileName + " to double");
      //	    utils::throwException(vals.size() == numPoints,
      //	     "Mismatch in number of points specified and number of points "
      //	     "provided in " + xPathInfo.xpath + " element in " +
      //	    xPathInfo.fileName);
      //	    const std::vector<std::pair<std::string, std::string>> attrString
      //= attrStrings[i]; 	    utils::throwException(attrString.size() == 3,
      //	    "Expected three attributes (i.e., n, l, m quantum numbers) in " +
      // xPathInfo.xpath + " element in " + xPathInfo.fileName);
      //	    utils::throwException(attrString[0].first=="n",
      //		"The first attribute for " + xPathInfo.xpath + " element in " +
      // xPathInfo.fileName + " must be \"n\"");
      //	    utils::throwException(attrString[1].first=="l",
      //		"The second attribute for " + xPathInfo.xpath + " element in " +
      // xPathInfo.fileName + " must be \"l\"");
      //	    utils::throwException(attrString[2].first=="m",
      //		"The third attribute for " + xPathInfo.xpath + " element in " +
      // xPathInfo.fileName + " must be \"m\""); 	    std::vector<int>
      // qNumbers(3); 	    for(size_type j = 0; j < 3; ++j)
      //	    {
      //	      convSuccess = utils::stringOps::strToInt(attrString[j].second,
      // qNumbers[j]); 	      utils::throwException(convSuccess,
      //                "Error while converting the " + std::to_string(j) +
      //		"-th quantum number in " + xPathInfo.xpath +
      //		" element in " + xPathInfo.fileName + " to integer");
      //	    }
      //	    d_qNumbersToRadialField[qNumbers] = vals;
      //	    qNumbersSet.insert(qNumbers);
      //	  }
      //
      //	  utils::throwException(qNumbersSet.size() == N,
      //	    "Found repeated quantum numbers while processing the attributes of
      //" + 	    xPathInfo.xpath + " element in " + xPathInfo.fileName);
      //
      //	}

      void
      processSphericalDataFromXMLNodeData(
        std::vector<std::shared_ptr<SphericalData>> &sphericalDataVec,
        const std::vector<std::string> &             radialValueStrings,
        const std::vector<std::string> &             qNumberStrings,
        const std::vector<std::string> &             cutOffInfoStrings,
        const std::vector<double> &                  radialPoints,
        const XPathInfo &                            xPathInfo,
        const SphericalHarmonicFunctions &           sphericalHarmonicFunc)
      {
        const size_type numPoints = radialPoints.size();
        const size_type N         = radialValueStrings.size();
        utils::throwException(
          N == qNumberStrings.size(),
          "Mismatch in number of \"values\" and \"qNumbers\" child "
          "elements found for element " +
            xPathInfo.xpath + " in file " + xPathInfo.fileName);
        utils::throwException(
          N == cutOffInfoStrings.size(),
          "Mismatch in number of \"values\" and \"cutoffInfo\" child "
          "elements found for element " +
            xPathInfo.xpath + " in file " + xPathInfo.fileName);

        sphericalDataVec.resize(N);

        // Make enum classes for analytial and numerical sphericalData ?

        std::set<std::vector<int>> qNumbersSet;
        bool                       convSuccess = false;
        for (size_type i = 0; i < N; ++i)
          {
            std::vector<double> radialValues(0);
            convSuccess =
              utils::stringOps::splitStringToDoubles(radialValueStrings[i],
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

            std::vector<int> qNumbers(0);
            convSuccess = utils::stringOps::splitStringToInts(qNumberStrings[i],
                                                              qNumbers,
                                                              3);
            utils::throwException(convSuccess,
                                  "Error while converting quantum numbers in " +
                                    xPathInfo.xpath + " element in " +
                                    xPathInfo.fileName + " to double");
            utils::throwException(
              qNumbers.size() == 3,
              "Expected three quantum number (i.e., n, l, m) in \"qNumbers\" "
              "child element in " +
                xPathInfo.xpath + " element in " + xPathInfo.fileName);
            int n = qNumbers[0];
            int l = qNumbers[1];
            int m = qNumbers[2];
            utils::throwException(
              n > 0,
              "Principal (n) quantum number less than 1 found in " +
                xPathInfo.xpath + " element in " + xPathInfo.fileName);
            utils::throwException(
              l < n,
              "Angular quantum number (l) greater than or equal to principal quantum "
              " (n) found in " +
                xPathInfo.xpath + " element in " + xPathInfo.fileName);
            utils::throwException(
              m >= -l && m <= l,
              "Magnetic quantum number (m) found outside of -l and +l "
              "(l = angular quantum number) in " +
                xPathInfo.xpath + " element in " + xPathInfo.fileName);
            qNumbersSet.insert(qNumbers);

            std::vector<double> cutoffInfo(0);
            convSuccess =
              utils::stringOps::splitStringToDoubles(cutOffInfoStrings[i],
                                                     cutoffInfo,
                                                     2);
            utils::throwException(convSuccess,
                                  "Error while converting cutoff info in " +
                                    xPathInfo.xpath + " element in " +
                                    xPathInfo.fileName + " to double");
            utils::throwException(
              cutoffInfo.size() == 2,
              "Expected two values (cutoff and smoothness factor) in "
              " \"cutoffInfo\" child element in " +
                xPathInfo.xpath + " element in " + xPathInfo.fileName);
            double cutoff     = cutoffInfo[0];
            double smoothness = cutoffInfo[1];

            sphericalDataVec[i] =
              std::make_shared<SphericalDataNumerical>(qNumbers,
                                                       radialPoints,
                                                       radialValues,
                                                       cutoff,
                                                       smoothness,
                                                       sphericalHarmonicFunc);
          }

        utils::throwException(
          qNumbersSet.size() == N,
          "Found repeated quantum numbers while processing " + xPathInfo.xpath +
            " element in " + xPathInfo.fileName);

        // Check if the m's for the quantum numbers are in ascending order.
        int i = 0;
        while (i < N)
          {
            auto it = qNumbersSet.begin();
            std::advance(it, i);
            int l     = *(it->data() + 1);
            int count = 0;
            for (int m = -l; m <= l; m++)
              {
                it = qNumbersSet.begin();
                std::advance(it, i + count);
                utils::throwException(
                  m == *(it->data() + 2),
                  "The quantum number m for " + xPathInfo.xpath +
                    " in the xml file " + xPathInfo.fileName +
                    " are not in correct order (ascending m).");
                count += 1;
              }
            i += count;
          }
      }

    } // namespace AtomSphericalDataXMLLocal

    namespace
    {
      std::string
      getXPath(const std::string &rootElementName,
               const std::string &ns,
               const std::string  elementName)
      {
        return "/" + ns + ":" + rootElementName + "/" + ns + ":" + elementName;
      }

      void
      getSphericalDataFromXMLNode(
        std::vector<std::shared_ptr<SphericalData>> &sphericalDataVec,
        const std::vector<double> &                  radialPoints,
        const AtomSphericalDataXMLLocal::XPathInfo & xPathInfo,
        const SphericalHarmonicFunctions &           sphericalHarmonicFunc)
      {
        std::vector<std::string> radialValuesStrings(0);
        std::vector<std::string> qNumbersStrings(0);
        std::vector<std::string> cutOffInfoStrings(0);
        AtomSphericalDataXMLLocal::readSphericalDataFromXMLNodeData(
          radialValuesStrings, qNumbersStrings, cutOffInfoStrings, xPathInfo);
        AtomSphericalDataXMLLocal::processSphericalDataFromXMLNodeData(
          sphericalDataVec,
          radialValuesStrings,
          qNumbersStrings,
          cutOffInfoStrings,
          radialPoints,
          xPathInfo,
          sphericalHarmonicFunc);
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

    AtomSphericalData::AtomSphericalData(
      const std::string                 fileName,
      const std::vector<std::string> &  fieldNames,
      const std::vector<std::string> &  metadataNames,
      const SphericalHarmonicFunctions &sphericalHarmonicFunc)
      : d_fileName(fileName)
      , d_fieldNames(fieldNames)
      , d_metadataNames(metadataNames)
    {
#if defined(LIBXML_XPATH_ENABLED) && defined(LIBXML_SAX1_ENABLED)
      xmlDocPtr   ptrToXmlDoc;
      std::string rootElementName = "atom";
      std::string ns              = "dft-efe";
      std::string nsHRef          = "http://www.dft-efe.com/dft-efe";
      ptrToXmlDoc                 = AtomSphericalDataXMLLocal::getDoc(fileName);

      AtomSphericalDataXMLLocal::XPathInfo xPathInfo;
      xPathInfo.fileName = fileName;
      xPathInfo.doc      = ptrToXmlDoc;
      xPathInfo.ns       = ns;
      xPathInfo.nsHRef   = nsHRef;

      std::vector<std::string> nodeStrings(0);
      //
      // storing meta data
      //
      for (size_type iMeta = 0; iMeta < metadataNames.size(); ++iMeta)
        {
          const std::string metadataName = metadataNames[iMeta];
          // get symbol
          xPathInfo.xpath = getXPath(rootElementName, ns, metadataName);
          AtomSphericalDataXMLLocal::getNodeStrings(xPathInfo, nodeStrings);
          utils::throwException(nodeStrings.size() == 1,
                                "Found more than one " + xPathInfo.xpath +
                                  " element in " + fileName);
          // remove leading or trailing whitespace
          d_metadata[metadataName] = utils::stringOps::trimCopy(nodeStrings[0]);
        }


      std::vector<std::vector<std::pair<std::string, std::string>>> attrStrings(
        0);

      bool convSuccess = false;

      // ---------------------This class constructor is only designed for
      // SphericalDataNumerical for now------------------------

      int                 numRadialPoints;
      std::vector<double> radialPoints;

      //
      // get number of radial points
      //
      xPathInfo.xpath = getXPath(rootElementName, ns, "NR");
      AtomSphericalDataXMLLocal::getNodeStrings(xPathInfo, nodeStrings);
      utils::throwException(nodeStrings.size() == 1,
                            "Found more than one " + xPathInfo.xpath +
                              " element in " + fileName);
      // remove leading or trailing whitespace
      utils::stringOps::trim(nodeStrings[0]);
      convSuccess = utils::stringOps::strToInt(nodeStrings[0], numRadialPoints);
      utils::throwException(convSuccess,
                            "Error while converting " + xPathInfo.xpath +
                              " element in " + fileName + " to int");
      utils::throwException(numRadialPoints > 0,
                            "Non-positive integer found for" + xPathInfo.xpath +
                              " element in " + fileName);

      // get radial points
      //
      xPathInfo.xpath = getXPath(rootElementName, ns, "r");
      AtomSphericalDataXMLLocal::getNodeStrings(xPathInfo, nodeStrings);
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
          xPathInfo.xpath = getXPath(rootElementName, ns, fieldName);
          std::vector<std::shared_ptr<SphericalData>> sphericalDataVec(0);
          std::map<std::vector<int>, size_type>       qNumbersToIdMap;
          getSphericalDataFromXMLNode(sphericalDataVec,
                                      radialPoints,
                                      xPathInfo,
                                      sphericalHarmonicFunc);
          storeQNumbersToDataIdMap(sphericalDataVec, qNumbersToIdMap);
          d_sphericalData[fieldName]   = sphericalDataVec;
          d_qNumbersToIdMap[fieldName] = qNumbersToIdMap;
        }

      xmlFreeDoc(ptrToXmlDoc);
      xmlCleanupParser();
#else
      utils::throwException(
        false, "Support for LIBXML XPATH and LIBXML SAX1 not found");
#endif // defined(LIBXML_XPATH_ENABLED) && defined(LIBXML_SAX1_ENABLED)
    }

    std::string
    AtomSphericalData::getFileName() const
    {
      return d_fileName;
    }

    std::vector<std::string>
    AtomSphericalData::getFieldNames() const
    {
      return d_fieldNames;
    }

    std::vector<std::string>
    AtomSphericalData::getMetadataNames() const
    {
      return d_metadataNames;
    }

    const std::vector<std::shared_ptr<SphericalData>> &
    AtomSphericalData::getSphericalData(const std::string fieldName) const
    {
      auto it = d_sphericalData.find(fieldName);
      DFTEFE_AssertWithMsg(it != d_sphericalData.end(),
                           ("FieldName " + fieldName +
                            " not while parsing the XML file:" + d_fileName)
                             .c_str());
      return it->second;
    }


    const std::shared_ptr<SphericalData>
    AtomSphericalData::getSphericalData(const std::string       fieldName,
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
    AtomSphericalData::getMetadata(const std::string metadataName) const
    {
      auto it = d_metadata.find(metadataName);
      utils::throwException(it != d_metadata.end(),
                            "Unable to find the metadata " + metadataName +
                              " while parsing the XML file " + d_fileName);

      return it->second;
    }


    size_type
    AtomSphericalData::getQNumberID(const std::string       fieldName,
                                    const std::vector<int> &qNumbers) const
    {
      auto it = d_qNumbersToIdMap.find(fieldName);
      utils::throwException<utils::InvalidArgument>(
        it != d_qNumbersToIdMap.end(),
        "Cannot find the atom symbol provided to AtomSphericalData::getQNumberID");
      auto it1 = (it->second).find(qNumbers);
      utils::throwException<utils::InvalidArgument>(
        it1 != (it->second).end(),
        "Cannot find the qnumbers provided to AtomSphericalData::getQNumberID");
      return (it1)->second;
    }

    size_type
    AtomSphericalData::nSphericalData(std::string fieldName) const
    {
      auto it = d_sphericalData.find(fieldName);
      utils::throwException<utils::InvalidArgument>(
        it != d_sphericalData.end(),
        "Cannot find the atom symbol provided to AtomSphericalData::nSphericalData");
      return (it->second).size();
    }

  } // end of namespace atoms
} // end of namespace dftefe
