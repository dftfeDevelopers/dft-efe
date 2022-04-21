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

#include <atoms/AtomSphericalElectronicData.h>
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

    namespace AtomSphericalElectronicDataXMLLocal 
    {

      struct XPathInfo 
      {
	xmlDocPtr doc;
	std::string filename;
	std::string xpath;
	std::string ns;
	std::string nsHRef;
      };
      
      bool
	splitStringToInts(const std::string s, std::vector<int> & vals, size_type reserveSize = 0)
	{
	  std::istringstream ss(s);
	  std::string word; 
	  size_type wordCount = 0;
	  vals.resize(0);
	  vals.reserve(reserveSize);
	  bool convSuccess = false; 
	  int x;
	  while (ss >> word) 
	  {
	    convSuccess = utils::stringOps::strToInt(word, x);
	    if(!convSuccess) break;
	    vals.push_back(x);
	  }
	  return convSuccess;
	}
      
      bool
	splitStringToDoubles(const std::string s, std::vector<double> & vals, size_type reserveSize = 0)
	{
	  std::istringstream ss(s);
	  std::string word; 
	  size_type wordCount = 0;
	  vals.resize(0);
	  vals.reserve(reserveSize);
	  bool convSuccess = false; 
	  double x;
	  while (ss >> word) 
	  {
	    convSuccess = utils::stringOps::strToDouble(word, x);
	    if(!convSuccess) break;
	    vals.push_back(x);
	  }
	  return convSuccess;
	}

      xmlDocPtr
	getDoc(const std::string filename) 
	{
	  xmlDocPtr doc;
	  doc = xmlParseFile(filename.c_str());
	  utils::throwException(doc != NULL, 
	      "Error parsing the XML file:" + filename);
	  return doc;
	}

      xmlXPathObjectPtr
	getNodeSet(xmlDocPtr doc, 
	    const std::string xpath, 
	    const std::string ns, 
	    const std::string nsHRef)
	{
	  xmlXPathContextPtr context;
	  xmlXPathObjectPtr result;
	  context = xmlXPathNewContext(doc);
	  utils::throwException(context != NULL, 
	      "Unable to create a new XPath context via xmlXPathNewContext()");

	  xmlChar * nsXmlChar = (xmlChar*)ns.c_str();
	  xmlChar * nsHRefXmlChar = (xmlChar*) nsHRef.c_str();
	  int returnRegNS = xmlXPathRegisterNs(context, BAD_CAST nsXmlChar, BAD_CAST nsHRefXmlChar);
	  utils::throwException(returnRegNS == 0, "Unable to register the with prefix " 
	      + ns + ":" + nsHRef);

	  xmlChar * xpathXmlChar = (xmlChar*) xpath.c_str();
	  result = xmlXPathEvalExpression(xpathXmlChar, context);
	  xmlXPathFreeContext(context);
	  utils::throwException(result != NULL, 
	      "Unable to evaluate the XPath expression " + xpath + 
	      " using xmlXPathEvalExpression");
	  if(xmlXPathNodeSetIsEmpty(result->nodesetval))
	  {
	    xmlXPathFreeObject(result);
	    utils::throwException(false, "XPath expression " + xpath + " not found.");
	  }
	  return result;
	}

      void
	getNodeStrings(const XPathInfo & xPathInfo,
	    std::vector<std::string> & nodeStrings)
	{
	  nodeStrings.resize(0);
	  xmlNodeSetPtr ptrToXmlNodeSet;
	  xmlXPathObjectPtr ptrToXmlXPathObject;
	  xmlChar *keyword;
	  ptrToXmlXPathObject = getNodeSet(xPathInfo.doc,
	      xPathInfo.xpath, 
	      xPathInfo.ns, 
	      xPathInfo.nsHRef);
	  ptrToXmlNodeSet = ptrToXmlXPathObject->nodesetval;
	  nodeStrings.resize(ptrToXmlNodeSet->nodeNr);
	  for(size_type i = 0; i < ptrToXmlNodeSet->nodeNr; i++) 
	  {
	    keyword = xmlNodeListGetString(xPathInfo.doc, ptrToXmlNodeSet->nodeTab[i]->xmlChildrenNode, 1);
	    nodeStrings[i] = std::string((char*)keyword);
	    xmlFree(keyword);
	  }
	  xmlXPathFreeObject(ptrToXmlXPathObject);
	}

      void
	getNodeStrings(const XPathInfo & xPathInfo,
	    std::vector<std::string> & nodeStrings,
	    std::vector<std::vector<std::pair<std::string, std::string>>> & attrStrings)
	{
	  nodeStrings.resize(0);
	  attrStrings.resize(0);
	  xmlNodeSetPtr ptrToXmlNodeSet;
	  xmlXPathObjectPtr ptrToXmlXPathObject;
	  xmlChar *keyword;
	  ptrToXmlXPathObject = getNodeSet(xPathInfo.doc,
	      xPathInfo.xpath, 
	      xPathInfo.ns, 
	      xPathInfo.nsHRef);
	  ptrToXmlNodeSet = ptrToXmlXPathObject->nodesetval;
	  nodeStrings.resize(ptrToXmlNodeSet->nodeNr);
	  attrStrings.resize(ptrToXmlNodeSet->nodeNr);
	  for(size_type i = 0; i < ptrToXmlNodeSet->nodeNr; i++) 
	  {
	    xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
	    keyword = xmlNodeListGetString(xPathInfo.doc, ptrToXmlNode->xmlChildrenNode, 1);
	    nodeStrings[i] = std::string((char*)keyword);
	    xmlFree(keyword);
	    // loop over attributes
	    xmlAttr* attribute = ptrToXmlNode->properties;
	    std::vector<std::pair<std::string, std::string>> & attrPairs =
	      attrStrings[i];
	    while(attribute)
	    {
	      const xmlChar *  attrName = attribute->name;
	      xmlChar * attrValue = xmlNodeListGetString(ptrToXmlNode->doc, attribute->children, 1);
	      std::string name((const char*)attrName);
	      std::string value((char*)attrValue); 
	      attrPairs.push_back(std::make_pair(name,value)); 
	      xmlFree(attrValue);
	      attribute = attribute->next;
	    }
	  }
	  xmlXPathFreeObject(ptrToXmlXPathObject);
	}

      void
	readSphericalDataFromXMLNodeData(
	    std::vector<std::string> & radialValuesStrings,
	    std::vector<std::string> & qNumbersStrings,
	    std::vector<std::string> & cutOffInfoStrings,
	    const XPathInfo & xPathInfo)
	{

	  radialValuesStrings.resize(0);
	  qNumbersStrings.resize(0);
	  cutOffInfoStrings.resize(0);
	  xmlNodeSetPtr ptrToXmlNodeSet;
	  xmlXPathObjectPtr ptrToXmlXPathObject;
	  xmlChar *keyword;
	  std::string xpathPrefix = xPathInfo.xpath + "/" + xPathInfo.ns + ":";
	  int numNodes = 0;
	  
	  //get the values
	  std::string valuesTag = "values";
	  ptrToXmlXPathObject = getNodeSet(xPathInfo.doc,
	      xpathPrefix + valuesTag, 
	      xPathInfo.ns, 
	      xPathInfo.nsHRef);
	  ptrToXmlNodeSet = ptrToXmlXPathObject->nodesetval;
	  numNodes = ptrToXmlNodeSet->nodeNr;
	  radialValuesStrings.resize(numNodes);
	  for(size_type i = 0; i < numNodes; i++) 
	  {
	    xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
	    keyword = xmlNodeListGetString(xPathInfo.doc, ptrToXmlNode->xmlChildrenNode, 1);
	    radialValuesStrings[i] = std::string((char*)keyword);
	    xmlFree(keyword);
	  }
	  xmlXPathFreeObject(ptrToXmlXPathObject);
	  
	  //get the qNumbers
	  std::string qNumbersTag = "qNumbers";
	  ptrToXmlXPathObject = getNodeSet(xPathInfo.doc,
	      xpathPrefix + qNumbersTag, 
	      xPathInfo.ns, 
	      xPathInfo.nsHRef);
	  ptrToXmlNodeSet = ptrToXmlXPathObject->nodesetval;
	  numNodes = ptrToXmlNodeSet->nodeNr;
	  qNumbersStrings.resize(numNodes);
	  for(size_type i = 0; i < numNodes; i++) 
	  {
	    xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
	    keyword = xmlNodeListGetString(xPathInfo.doc, ptrToXmlNode->xmlChildrenNode, 1);
	    qNumbersStrings[i] = std::string((char*)keyword);
	    xmlFree(keyword);
	  }
	  xmlXPathFreeObject(ptrToXmlXPathObject);
	  
	  //get the cutoff infor 
	  std::string cutOffInfoTag = "cutoffInfo";
	  ptrToXmlXPathObject = getNodeSet(xPathInfo.doc,
	      xpathPrefix + cutOffInfoTag, 
	      xPathInfo.ns, 
	      xPathInfo.nsHRef);
	  ptrToXmlNodeSet = ptrToXmlXPathObject->nodesetval;
	  numNodes = ptrToXmlNodeSet->nodeNr;
	  cutOffInfoStrings.resize(numNodes);
	  for(size_type i = 0; i < numNodes; i++) 
	  {
	    xmlNodePtr ptrToXmlNode = ptrToXmlNodeSet->nodeTab[i];
	    keyword = xmlNodeListGetString(xPathInfo.doc, ptrToXmlNode->xmlChildrenNode, 1);
	    cutOffInfoStrings[i] = std::string((char*)keyword);
	    xmlFree(keyword);
	  }
	  xmlXPathFreeObject(ptrToXmlXPathObject);
	}
      
//      void
//	processSphericalDataFromXMLNodeData(std::map<std::vector<int>, std::vector<double>> & d_qNumbersToRadialField,
//	    const std::vector<std::string> & nodeStrings,
//	    const std::vector<std::vector<std::pair<std::string, std::string>>> & attrStrings,
//	    const XPathInfo & xPathInfo,
//	    const size_type numPoints)
//	{
//
//	  const size_type N  = nodeStrings.size();
//	  std::set<std::vector<int>> qNumbersSet;
//	  bool convSuccess = false;
//	  for(size_type i = 0; i < N; ++i)
//	  {
//	    std::vector<double> vals(0);
//	    convSuccess = splitStringToDoubles(nodeStrings[0], vals, numPoints); 
//      	    utils::throwException(convSuccess, 
//	  	"Error while converting values in " + xPathInfo.xpath + " element in " + xPathInfo.filename + " to double");
//	    utils::throwException(vals.size() == numPoints, 
//	     "Mismatch in number of points specified and number of points "
//	     "provided in " + xPathInfo.xpath + " element in " + 
//	    xPathInfo.filename);
//	    const std::vector<std::pair<std::string, std::string>> attrString = attrStrings[i];
//	    utils::throwException(attrString.size() == 3, 
//	    "Expected three attributes (i.e., n, l, m quantum numbers) in " + xPathInfo.xpath + " element in " + xPathInfo.filename);
//	    utils::throwException(attrString[0].first=="n", 
//		"The first attribute for " + xPathInfo.xpath + " element in " + xPathInfo.filename + " must be \"n\""); 
//	    utils::throwException(attrString[1].first=="l", 
//		"The second attribute for " + xPathInfo.xpath + " element in " + xPathInfo.filename + " must be \"l\""); 
//	    utils::throwException(attrString[2].first=="m", 
//		"The third attribute for " + xPathInfo.xpath + " element in " + xPathInfo.filename + " must be \"m\"");
//	    std::vector<int> qNumbers(3);
//	    for(size_type j = 0; j < 3; ++j)
//	    {
//	      convSuccess = utils::stringOps::strToInt(attrString[j].second, qNumbers[j]);
//	      utils::throwException(convSuccess,
//                "Error while converting the " + std::to_string(j) + 
//		"-th quantum number in " + xPathInfo.xpath + 
//		" element in " + xPathInfo.filename + " to integer");
//	    } 
//	    d_qNumbersToRadialField[qNumbers] = vals;
//	    qNumbersSet.insert(qNumbers);
//	  }
//		
//	  utils::throwException(qNumbersSet.size() == N, 
//	    "Found repeated quantum numbers while processing the attributes of " + 
//	    xPathInfo.xpath + " element in " + xPathInfo.filename);
//
//	}
      
      void
	processSphericalDataFromXMLNodeData(std::vector<SphericalData> & sphericalDataVec,
	    const std::vector<std::string> & radialValueStrings,
	    const std::vector<std::string> & qNumberStrings,
	    const std::vector<std::string> & cutOffInfoStrings,
	    const std::vector<double> & radialPoints,
	    const XPathInfo & xPathInfo)
	{

	  const size_type numPoints = radialPoints.size();
	  const size_type N  = radialValueStrings.size();
	  utils::throwException(N == qNumberStrings.size(), 
	      "Mismatch in number of \"values\" and \"qNumbers\" child "
	      "elements found for element " + xPathInfo.xpath + " in file "
	      + xPathInfo.filename);
	  utils::throwException(N == cutOffInfoStrings.size(), 
	      "Mismatch in number of \"values\" and \"cutoffInfo\" child "
	      "elements found for element " + xPathInfo.xpath  + " in file "
	      + xPathInfo.filename);

	  sphericalDataVec.resize(N);
	
	  std::set<std::vector<int>> qNumbersSet;
	  bool convSuccess = false;
	  for(size_type i = 0; i < N; ++i)
	  {
	    SphericalData & sphericalData = sphericalDataVec[i];
	    sphericalData.radialPoints = radialPoints;
	    
	    convSuccess = splitStringToDoubles(radialValueStrings[i], 
		sphericalData.radialValues, 
		numPoints); 
      	    utils::throwException(convSuccess, 
	  	"Error while converting values in " + xPathInfo.xpath + " element in " + xPathInfo.filename + " to double");
	    utils::throwException(sphericalData.radialValues.size() == numPoints, 
	     "Mismatch in number of points specified and number of points "
	     "provided in " + xPathInfo.xpath + " element in " + 
	    xPathInfo.filename);

	    convSuccess = splitStringToInts(qNumberStrings[i], 
		sphericalData.qNumbers, 3); 
      	    utils::throwException(convSuccess, 
	  	"Error while converting quantum numbers in " 
		+ xPathInfo.xpath + " element in " + xPathInfo.filename + 
		" to double");
	    utils::throwException(sphericalData.qNumbers.size() == 3, 
	    "Expected three quantum number (i.e., n, l, m) in \"qNumbers\" "
	    "child element in " + xPathInfo.xpath + " element in " + 
	    xPathInfo.filename);
	    int n = sphericalData.qNumbers[0];
	    int l = sphericalData.qNumbers[1];
	    int m = sphericalData.qNumbers[2];
	    utils::throwException(n > 0, 
		"Principal (n) quantum number less than 1 found in " + 
		xPathInfo.xpath + " element in " + xPathInfo.filename);
	    utils::throwException(l < n, 
		"Angular quantum number (l) greater than or equal to principal quantum "
		" (n) found in " + xPathInfo.xpath + " element in " 
		+ xPathInfo.filename); 
	    utils::throwException(m >= -l && m <= l, 
		"Magnetic quantum number (m) found outside of -l and +l "
		"(l = angular quantum number) in " + xPathInfo.xpath + 
		" element in " + xPathInfo.filename);
	    qNumbersSet.insert(sphericalData.qNumbers);
	   
	    std::vector<double> cutoffInfo(0);
	    convSuccess = splitStringToDoubles(cutOffInfoStrings[i], 
		cutoffInfo, 
		2); 
      	    utils::throwException(convSuccess, 
	  	"Error while converting cutoff info in " + xPathInfo.xpath + 
		" element in " + xPathInfo.filename + " to double");
	    utils::throwException(cutoffInfo.size() == 2, 
	     "Expected two values (cutoff and smoothness factor) in " 
	     " \"cutoffInfo\" child element in " + xPathInfo.xpath + 
	     " element in " + xPathInfo.filename);
	    sphericalData.cutoff = cutoffInfo[0];
	    sphericalData.smoothness = cutoffInfo[1];
	  }
		
	  utils::throwException(qNumbersSet.size() == N, 
	    "Found repeated quantum numbers while processing " + 
	    xPathInfo.xpath + " element in " + xPathInfo.filename);

	}

    }

    namespace {


      std::string getXPath(const std::string & rootElementName,
	  const std::string & ns,
	  const std::string elementName)
      {
	return "/" + ns + ":" + rootElementName + "/" + ns + ":" + elementName;
      }
      
      void getSphericalDataFromXMLNode(std::vector<SphericalData> & sphericalDataVec,
	  const std::vector<double> & radialPoints,
	  const AtomSphericalElectronicDataXMLLocal::XPathInfo & xPathInfo)
      {
	std::vector<std::string> radialValuesStrings(0);
	std::vector<std::string> qNumbersStrings(0);
	std::vector<std::string> cutOffInfoStrings(0);
	AtomSphericalElectronicDataXMLLocal::readSphericalDataFromXMLNodeData(radialValuesStrings, qNumbersStrings, cutOffInfoStrings, xPathInfo);
	AtomSphericalElectronicDataXMLLocal::processSphericalDataFromXMLNodeData(sphericalDataVec,
	    radialValuesStrings,
	    qNumbersStrings,
	    cutOffInfoStrings,
	    radialPoints,
	    xPathInfo);
      }

      void storeQNumbersToDataIdMap(const std::vector<SphericalData> & sphericalDataVec,
	  std::map<std::vector<int>, size_type> & qNumbersToDataIdMap)
      {
	size_type N = sphericalDataVec.size();
	for(size_type i = 0; i < N; ++i)
	{
	  qNumbersToDataIdMap[sphericalDataVec[i].qNumbers] = i;
	}
      }


    }

    AtomSphericalElectronicData::AtomSphericalElectronicData(const std::string filename):
      d_filename(filename),
      d_charge(0.0),
      d_Z(0.0)
    {
#if defined(LIBXML_XPATH_ENABLED) && defined(LIBXML_SAX1_ENABLED)
      xmlDocPtr ptrToXmlDoc;
      std::string rootElementName = "atom";
      std::string ns = "dft-efe";
      std::string nsHRef = "http://www.dft-efe.com/dft-efe";
      ptrToXmlDoc = AtomSphericalElectronicDataXMLLocal::getDoc(filename);

      AtomSphericalElectronicDataXMLLocal::XPathInfo xPathInfo;
      xPathInfo.filename = filename;
      xPathInfo.doc = ptrToXmlDoc;
      xPathInfo.ns = ns;
      xPathInfo.nsHRef = nsHRef;

      bool convSuccess = false;
      std::vector<std::string> nodeStrings(0);
      std::vector<std::vector<std::pair<std::string, std::string>>> attrStrings(0);

      // get symbol
      xPathInfo.xpath = getXPath(rootElementName, ns, "symbol");
      AtomSphericalElectronicDataXMLLocal::getNodeStrings(xPathInfo, nodeStrings);
      utils::throwException(nodeStrings.size()==1, 
	  "Found more than one " + xPathInfo.xpath  + " element in " + filename);
      // remove leading or trailing whitespace
      d_symbol = utils::stringOps::trimCopy(nodeStrings[0]);

      // get atomic number
      xPathInfo.xpath = getXPath(rootElementName, ns, "Z");
      AtomSphericalElectronicDataXMLLocal::getNodeStrings(xPathInfo, nodeStrings);
      utils::throwException(nodeStrings.size()==1, 
	  "Found more than one " + xPathInfo.xpath  + " element in " + filename);
      // remove leading or trailing whitespace
      utils::stringOps::trim(nodeStrings[0]);
      convSuccess = utils::stringOps::strToDouble(nodeStrings[0], d_Z);
      utils::throwException(convSuccess, 
	  "Error while converting " + xPathInfo.xpath + " element in " + filename + " to double");

      // get charge
      xPathInfo.xpath = getXPath(rootElementName, ns, "charge");
      AtomSphericalElectronicDataXMLLocal::getNodeStrings(xPathInfo, nodeStrings);
      utils::throwException(nodeStrings.size()==1, 
	  "Found more than one " + xPathInfo.xpath  + " element in " + filename);
      // remove leading or trailing whitespace
      utils::stringOps::trim(nodeStrings[0]);
      convSuccess = utils::stringOps::strToDouble(nodeStrings[0], d_charge);
      utils::throwException(convSuccess, 
	  "Error while converting " + xPathInfo.xpath + " element in " + filename + " to double");

      // get number of radial points 
      xPathInfo.xpath = getXPath(rootElementName, ns, "NR");;
      AtomSphericalElectronicDataXMLLocal::getNodeStrings(xPathInfo, nodeStrings);
      utils::throwException(nodeStrings.size()==1, 
	  "Found more than one " + xPathInfo.xpath  + " element in " + filename);
      // remove leading or trailing whitespace
      utils::stringOps::trim(nodeStrings[0]);
      int numRadialPoints;
      convSuccess = utils::stringOps::strToInt(nodeStrings[0], numRadialPoints);
      utils::throwException(convSuccess, 
	  "Error while converting " + xPathInfo.xpath + " element in " + filename + " to int");
      utils::throwException(numRadialPoints > 0, 
	  "Negative integer found for" + xPathInfo.xpath + " element in " + filename);
      d_numRadialPoints = (size_type) numRadialPoints;

      // get radial points
      xPathInfo.xpath = getXPath(rootElementName, ns, "r");;
      AtomSphericalElectronicDataXMLLocal::getNodeStrings(xPathInfo, nodeStrings);
      utils::throwException(nodeStrings.size()==1, 
	  "Found more than one " + xPathInfo.xpath  + " element in " + filename);
      d_radialPoints.resize(0); 
      convSuccess = AtomSphericalElectronicDataXMLLocal::splitStringToDoubles(nodeStrings[0], d_radialPoints, d_numRadialPoints); 
      utils::throwException(convSuccess, 
	  "Error while converting " + xPathInfo.xpath + " element in " + filename + " to double");
      utils::throwException(d_radialPoints.size()==d_numRadialPoints,
	  "Mismatch in number of radial points specified and the number of "
	  " radial points provided in " + filename);

      // get density values
      xPathInfo.xpath = getXPath(rootElementName, ns, "density");
      getSphericalDataFromXMLNode(d_densityData, d_radialPoints, xPathInfo);
      storeQNumbersToDataIdMap(d_densityData, d_qNumbersToDensityDataIdMap);

      // get Hartree potential values
      xPathInfo.xpath = getXPath(rootElementName, ns, "vhartree");
      getSphericalDataFromXMLNode(d_vHartreeData, d_radialPoints, xPathInfo);
      storeQNumbersToDataIdMap(d_vHartreeData, d_qNumbersToVHartreeDataIdMap);
      
      // get nuclear potential values
      xPathInfo.xpath = getXPath(rootElementName, ns, "vnuclear");
      getSphericalDataFromXMLNode(d_vNuclearData, d_radialPoints, xPathInfo);
      storeQNumbersToDataIdMap(d_vNuclearData, d_qNumbersToVNuclearDataIdMap);
      utils::throwException(d_vHartreeData.size() == d_vNuclearData.size(),
	  "Mismatch in number of total hartree and nuclear potential found in file: " 
	  + d_filename);
      
      // get total potential values (total potential = hartree potential + nuclear potential)
      xPathInfo.xpath = getXPath(rootElementName, ns, "vtotal");
      getSphericalDataFromXMLNode(d_vTotalData, d_radialPoints, xPathInfo);
      storeQNumbersToDataIdMap(d_vTotalData, d_qNumbersToVTotalDataIdMap);
    
      utils::throwException(d_vTotalData.size() == d_vNuclearData.size(),
	  "Mismatch in number of total potential and nuclear potential found in file: " 
	  + d_filename);

      // get orbitals
      xPathInfo.xpath = getXPath(rootElementName, ns, "orbital");
      getSphericalDataFromXMLNode(d_orbitalData, d_radialPoints, xPathInfo);
      storeQNumbersToDataIdMap(d_orbitalData, d_qNumbersToOrbitalDataIdMap);

      xmlFreeDoc(ptrToXmlDoc);
      xmlCleanupParser();
#else
      utils::throwException(false, "Support for LIBXML XPATH and LIBXML SAX1 not found");
#endif // defined(LIBXML_XPATH_ENABLED) && defined(LIBXML_SAX1_ENABLED)
    }

    double
      AtomSphericalElectronicData::getAtomicNumber() const
      {
	return d_Z;
      }

    double
      AtomSphericalElectronicData::getCharge() const
      {
	return d_charge;
      }

    std::string
      AtomSphericalElectronicData::getSymbol() const
      {
	return d_symbol;
      }

    std::vector<double> 
      AtomSphericalElectronicData::getRadialPoints() const
      {
	return d_radialPoints;
      }
    
    const std::vector<SphericalData> & 
      AtomSphericalElectronicData::getDensityData() const
      {
	return d_densityData;
      }

    const std::vector<SphericalData> & 
      AtomSphericalElectronicData::getVHartreeData() const
      {
	return d_vHartreeData;
      }
    
    const std::vector<SphericalData> & 
      AtomSphericalElectronicData::getVNuclearData() const
      {
	return d_vNuclearData;
      }
    
    const std::vector<SphericalData> & 
      AtomSphericalElectronicData::getVTotalData() const
      {
	return d_vTotalData;
      }
    
    const std::vector<SphericalData> & 
      AtomSphericalElectronicData::getOrbitalData() const
      {
	return d_orbitalData;
      }

  } // end of namespace atoms
} // end of namespace dftefe
