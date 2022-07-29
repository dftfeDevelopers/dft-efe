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
#include <atoms/AtomFieldDataSpherical.h>
#include <utils/StringOperations.h>
#include <utils/Exceptions.h>
#include <fstream>
#include <sstream>
namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      void
      readRadialPoints(std::vector<double> &radialGridPoints,
                       const std::string    filename,
                       const std::string    radialPointsString)
      {
        std::string        word;
        std::istringstream keyString(
          utils::stringOps::trimCopy(radialPointsString));
        std::vector<std::string> wordsInKey(0);
        while (keyString >> word)
          {
            wordsInKey.push_back(utils::stringOps::trimCopy(word));
          }
        size_type     numWordsInKey = wordsInKey.size();
        std::ifstream readfile;
        std::string   readline;
        readfile.open(filename.c_str());
        utils::throwException(readfile.is_open(),
                              "File " + filename + " not found.");
        bool found = false;
        while (std::getline(readfile, readline))
          {
            if (readline[0] != '#')
              {
                if (readline.find(radialPointsString) != std::string::npos)
                  {
                    found = true;
                    break;
                  }
              }
          }

        std::string msg = "The keyword: " + radialPointsString +
                          " not found in the file " + filename;
        utils::throwException(found, msg);

        // remove leading and trailing whitespaces
        utils::stringOps::trim(readline);
        std::istringstream lineString(readline);
        size_type          wordCount = 0;
        int                nPoints   = 0;
        while (lineString >> word)
          {
            // remove leading and trailing whitespaces
            utils::stringOps::trim(word);
            if (word[0] == '#')
              break;
            if (wordCount < numWordsInKey)
              {
                std::string msg = "The " + std::to_string(wordCount) +
                                  " word in line "
                                  "containing the radial grid points in file " +
                                  filename + " should be " +
                                  wordsInKey[wordCount];
                utils::throwException(word == wordsInKey[wordCount], msg);
              }
            if (wordCount == numWordsInKey)
              {
                bool success = utils::stringOps::strToInt(word, nPoints);
                utils::throwException(
                  success && nPoints >= 0,
                  "The number of radial grid points must be a non-negative integer. Received value:" +
                    word);
              }

            wordCount++;
          }

        msg = "In file " + filename +
              ", undefined behavior in the line containing the keyword " +
              radialPointsString +
              ". It is expected to contain only two values: the " +
              radialPointsString + " keyword and the " + "number of points.";
        utils::throwException(wordCount == numWordsInKey + 1, msg);

        found = false;
        while (!found)
          {
            if (std::getline(readfile, readline))
              {
                utils::stringOps::trim(readline);
                if (readline[0] != '#')
                  {
                    if (!readline.empty())
                      {
                        found = true;
                      }
                    else
                      utils::throwException(false,
                                            "Empty line detected in file " +
                                              filename);
                  }
              }
            else
              utils::throwException(false,
                                    "Unexpected end of line reached in file " +
                                      filename);
          }

        wordCount = 0;
        radialGridPoints.resize(nPoints, 0.0);
        std::istringstream lineStringPoints(readline);
        bool               conversionSuccess;
        double             x;
        while (lineStringPoints >> word)
          {
            // remove leading and trailing whitespaces
            utils::stringOps::trim(word);
            if (word[0] == '#')
              break;
            conversionSuccess = utils::stringOps::strToDouble(word, x);
            utils::throwException(
              conversionSuccess,
              "Expected a number (double datatype) for the radial grid points. Received value: " +
                word);
            radialGridPoints[wordCount] = x;
            wordCount++;
          }

        msg = "Expected number of radial grid points in " + filename + " is " +
              std::to_string(nPoints) + ". But found " +
              std::to_string(wordCount) + " points.";
        utils::throwException(nPoints == wordCount, msg);

        readfile.close();
      }

      void
      readSphericalFieldData(std::vector<int> &   quantumNumbers,
                             double &             rcut,
                             double &             smoothness,
                             std::vector<double> &values,
                             const std::string    attributesLine,
                             const std::string    dataLine)
      {
        // remove leading and trailing whitespaces
        std::string attributesLineTrimmed =
          utils::stringOps::trimCopy(attributesLine);
        std::istringstream attributesLineString(attributesLineTrimmed);
        std::string        word;
        size_type          wordCount = 0;
        int                nPoints   = 0;
        quantumNumbers.resize(3, 0);
        values.resize(0);
        bool conversionSuccess;
        while (attributesLineString >> word)
          {
            // remove leading and trailing whitespaces
            utils::stringOps::trim(word);
            if (word[0] == '#')
              break;
            if (wordCount == 0)
              {
                conversionSuccess =
                  utils::stringOps::strToInt(word, quantumNumbers[0]);
                utils::throwException(
                  conversionSuccess,
                  "The n quantum number must be an integer. Received value: " +
                    word);
              }
            if (wordCount == 1)
              {
                conversionSuccess =
                  utils::stringOps::strToInt(word, quantumNumbers[1]);
                utils::throwException(
                  conversionSuccess,
                  "The l quantum number must be an integer. Received value: " +
                    word);
              }
            if (wordCount == 2)
              {
                conversionSuccess =
                  utils::stringOps::strToInt(word, quantumNumbers[2]);
                utils::throwException(
                  conversionSuccess,
                  "The m quantum number must be an integer. Received value: " +
                    word);
              }
            if (wordCount == 3)
              {
                conversionSuccess = utils::stringOps::strToDouble(word, rcut);
                utils::throwException(
                  conversionSuccess,
                  "The rcut must be a number (double datatype). Received value: " +
                    word);
              }
            if (wordCount == 4)
              {
                conversionSuccess =
                  utils::stringOps::strToDouble(word, smoothness);
                utils::throwException(
                  conversionSuccess,
                  "The smoothness factor must be a number (double datatype). Received value: " +
                    word);
              }
            if (wordCount == 5)
              {
                conversionSuccess = utils::stringOps::strToInt(word, nPoints);
                utils::throwException(
                  conversionSuccess && nPoints >= 0,
                  "The number of points in radial function must be non-negative integer. Received value: " +
                    word);
              }
            wordCount++;
          }

        std::string msg =
          "Undefined behavior in the line containing the field attributes. "
          "It is expected to contain only six values: n quantum number, l quantum number, m quantum number, "
          "rcut, smoothness, and number of points.";
        utils::throwException(wordCount == 6, msg);

        std::string dataLineTrimmed = utils::stringOps::trimCopy(dataLine);
        std::istringstream dataLineString(dataLineTrimmed);
        wordCount = 0;
        values.resize(nPoints, 0.0);
        double x;
        while (dataLineString >> word)
          {
            // remove leading and trailing whitespaces
            utils::stringOps::trim(word);
            if (word[0] == '#')
              break;
            conversionSuccess = utils::stringOps::strToDouble(word, x);
            utils::throwException(
              conversionSuccess,
              "The radial function value must be a number (double datatype). Received value: " +
                word);
            values[wordCount] = x;
            wordCount++;
          }

        msg = "Expected number of points in line " + dataLine + " is " +
              std::to_string(nPoints) + ". But found " +
              std::to_string(wordCount) + " points.";
        utils::throwException(nPoints == wordCount, msg);
      }


      void
      readFieldData(std::vector<std::vector<int>> &   quantumNumbersVec,
                    std::vector<double> &             radialGridPoints,
                    std::vector<std::vector<double>> &radialFunctionValues,
                    std::vector<double> &             rcutVec,
                    std::vector<double> &             smoothnessVec,
                    const std::string                 filename,
                    const std::string                 fieldName,
                    const std::string                 radialPointsString)
      {
        std::string        word;
        std::istringstream keyString(utils::stringOps::trimCopy(fieldName));
        std::vector<std::string> wordsInKey(0);
        while (keyString >> word)
          {
            wordsInKey.push_back(utils::stringOps::trimCopy(word));
          }
        size_type numWordsInKey = wordsInKey.size();

        // read radial grid points
        readRadialPoints(radialGridPoints, filename, radialPointsString);

        std::ifstream readfile;
        std::string   readline;
        readfile.open(filename.c_str());
        utils::throwException(readfile.is_open(),
                              "File " + filename + " not found.");
        bool found = false;
        while (std::getline(readfile, readline))
          {
            if (readline[0] != '#')
              {
                if (readline.find(fieldName) != std::string::npos)
                  {
                    found = true;
                    break;
                  }
              }
          }

        std::string msg =
          "Keyword: " + fieldName + " not found in the file " + filename;
        utils::throwException(found, msg);

        std::istringstream lineString(readline);
        size_type          wordCount = 0;
        int                nFields   = 0;
        while (lineString >> word)
          {
            // remove leading and trailing whitespaces
            utils::stringOps::trim(word);
            if (word[0] == '#')
              break;
            if (wordCount < numWordsInKey)
              {
                std::string msg = "The " + std::to_string(wordCount) +
                                  " word in line containing the keyword " +
                                  fieldName + " in file " + filename +
                                  " should be " + wordsInKey[wordCount];
                utils::throwException(word == wordsInKey[wordCount], msg);
              }
            if (wordCount == numWordsInKey)
              {
                bool conversionSuccess =
                  utils::stringOps::strToInt(word, nFields);
                utils::throwException(
                  conversionSuccess && nFields >= 0,
                  "The number of fields must be a non-negative integer. Received value: " +
                    word);
              }
            wordCount++;
          }

        msg =
          "Undefined behavior in the line containing the keyword " + fieldName +
          ". It is expected to contain two values: keyword and number of fields.";
        utils::throwException(wordCount == numWordsInKey + 1, msg);

        quantumNumbersVec.resize(nFields);
        radialFunctionValues.resize(nFields);
        rcutVec.resize(nFields);
        smoothnessVec.resize(nFields);
        bool                     allFieldsTraversed = false;
        size_type                fieldCount         = 0;
        std::vector<std::string> lineStrings(0);
        while (!allFieldsTraversed)
          {
            if (std::getline(readfile, readline))
              {
                utils::stringOps::trim(readline);
                if (readline[0] != '#')
                  {
                    if (!readline.empty())
                      {
                        lineStrings.push_back(readline);
                        fieldCount++;
                      }
                    else
                      utils::throwException(false,
                                            "Empty line detected in file " +
                                              filename);
                  }
              }
            else
              utils::throwException(false,
                                    "Unexpected end of line reached in file " +
                                      filename);

            if (fieldCount == 2 * nFields)
              allFieldsTraversed = true;
          }

        for (size_type iField = 0; iField < nFields; ++iField)
          {
            readSphericalFieldData(quantumNumbersVec[iField],
                                   rcutVec[iField],
                                   smoothnessVec[iField],
                                   radialFunctionValues[iField],
                                   lineStrings[2 * iField],
                                   lineStrings[2 * iField + 1]);
            std::string msg =
              "In file " + filename +
              " found mismatch in size of radial grid (size = " +
              std::to_string(radialGridPoints.size()) +
              ") and radial function (size = " +
              std::to_string(radialFunctionValues[iField].size()) + ")";
            utils::throwException(radialGridPoints.size() ==
                                    radialFunctionValues[iField].size(),
                                  msg);
          }
        readfile.close();
      }
    } // namespace


    AtomFieldDataSpherical::AtomFieldDataSpherical(
      const std::string filename,
      const std::string atomFieldname)
      : d_filename(filename)
      , d_atomFieldname(atomFieldname)
    {
      d_radialGridPoints.resize(1);
      d_radialFunctionValues.resize(0);
      d_quantumNumbers.resize(0);
      std::vector<double> rcutVec(0);
      std::vector<double> smoothnessVec(0);

      const std::string radialPointsString = "Radial points";
      readFieldData(d_quantumNumbers,
                    d_radialGridPoints[0],
                    d_radialFunctionValues,
                    rcutVec,
                    smoothnessVec,
                    filename,
                    atomFieldname,
                    radialPointsString);

      const size_type nFields = d_quantumNumbers.size();
      //
      // NOTE: Using the same radial grid points for all the fields. Hence the
      // radial grid Id is set to zero
      //
      size_type radialGridId = 0;
      for (size_type iField = 0; iField < nFields; ++iField)
        {
          d_quantumNumbersToRadialGridAndFunctionIdMap
            [d_quantumNumbers[iField]] = std::make_pair(radialGridId, iField);

          d_quantumNumbersToCutoffRadiusAndSmoothnessMap
            [d_quantumNumbers[iField]] =
              std::make_pair(rcutVec[iField], smoothnessVec[iField]);
        }
    }

    std::vector<std::vector<int>>
    AtomFieldDataSpherical::getQuantumNumbers() const
    {
      return d_quantumNumbers;
    }

    std::vector<double>
    AtomFieldDataSpherical::getRadialGridPoints(
      const std::vector<int> &quantumNumbers) const
    {
      auto it =
        d_quantumNumbersToRadialGridAndFunctionIdMap.find(quantumNumbers);
      if (it == d_quantumNumbersToRadialGridAndFunctionIdMap.end())
        {
          std::string quantumNumbersStr = "";
          for (size_type i = 0; i < quantumNumbers.size(); ++i)
            quantumNumbersStr += std::to_string(quantumNumbers[i]);
          std::string msg = "The given quantum numbers: " + quantumNumbersStr +
                            " for field " + d_atomFieldname +
                            " is not found in file " + d_filename;
          utils::throwException(false, msg);
        }

      size_type radialGridId = (it->second).first;
      return d_radialGridPoints[radialGridId];
    }

    std::pair<std::vector<double>, std::vector<double>>
    AtomFieldDataSpherical::getRadialFunction(
      const std::vector<int> &quantumNumbers) const
    {
      auto it =
        d_quantumNumbersToRadialGridAndFunctionIdMap.find(quantumNumbers);
      if (it == d_quantumNumbersToRadialGridAndFunctionIdMap.end())
        {
          std::string quantumNumbersStr = "";
          for (size_type i = 0; i < quantumNumbers.size(); ++i)
            quantumNumbersStr += std::to_string(quantumNumbers[i]);
          std::string msg = "The given quantum numbers: " + quantumNumbersStr +
                            " for field " + d_atomFieldname +
                            " is not found in file " + d_filename;
          utils::throwException(false, msg);
          size_type radialGridId = (it->second).first;
          size_type functionId   = (it->second).second;
          return std::make_pair(d_radialGridPoints[radialGridId],
                                d_radialFunctionValues[functionId]);
        }

      size_type radialGridId = (it->second).first;
      size_type functionId   = (it->second).second;
      return std::make_pair(d_radialGridPoints[radialGridId],
                            d_radialFunctionValues[functionId]);
    }

    std::pair<double, double>
    AtomFieldDataSpherical::getCutOffAndSmoothness(
      const std::vector<int> &quantumNumbers) const
    {
      auto it =
        d_quantumNumbersToCutoffRadiusAndSmoothnessMap.find(quantumNumbers);
      if (it == d_quantumNumbersToCutoffRadiusAndSmoothnessMap.end())
        {
          std::string quantumNumbersStr = "";
          for (size_type i = 0; i < quantumNumbers.size(); ++i)
            quantumNumbersStr += std::to_string(quantumNumbers[i]);
          std::string msg = "The given quantum numbers: " + quantumNumbersStr +
                            " for field " + d_atomFieldname +
                            " is not found in file " + d_filename;
          utils::throwException(false, msg);
        }
      return std::make_pair((it->second).first, (it->second).second);
    }
  } // end of namespace atoms
} // end of namespace dftefe
