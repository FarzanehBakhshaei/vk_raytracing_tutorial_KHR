#include <SimVisData.hpp>

#include <string>
#include <fstream>
#include <sstream>


SimVisDataPtr SimVisData::loadFromFile(const char* filepath)
{
	SimVisDataPtr simVisDataPtr = nullptr;
    std::vector<nvmath::vec3f> verts;
    std::vector<int> cellInds;
    std::vector<float> attributeList;
  
    std::ifstream in(filepath);
    if (!in.good())
        return nullptr;

    uint32_t numCellsX, numCellsY, numCellsZ;
    in >> numCellsX;
    in >> numCellsY;
    in >> numCellsZ;

    uint32_t numCellsTotal = numCellsX * numCellsY * numCellsZ;
    
    // add the vertices.
    for (uint32_t z = 0; z < numCellsZ + 1; z++) {
        for (uint32_t y = 0; y < numCellsY + 1; y++) {
            for (uint32_t x = 0; x < numCellsX + 1; x++) {
                verts.push_back(nvmath::vec3f(x, y, z));
            }
        }
    }

    // add the cell ids. Used if we deal with unstructure grid
    for (uint32_t z = 0; z < numCellsZ; z++) {
        for (uint32_t y = 0; y < numCellsY; y++) {
            for (uint32_t x = 0; x < numCellsX; x++) {
                cellInds.push_back(PT_IDXn(x + 0, y + 0, z + 0));
                cellInds.push_back(PT_IDXn(x + 1, y + 0, z + 0));
                cellInds.push_back(PT_IDXn(x + 1, y + 1, z + 0));
                cellInds.push_back(PT_IDXn(x + 0, y + 1, z + 0));
                cellInds.push_back(PT_IDXn(x + 0, y + 0, z + 1));
                cellInds.push_back(PT_IDXn(x + 1, y + 0, z + 1));
                cellInds.push_back(PT_IDXn(x + 1, y + 1, z + 1));
                cellInds.push_back(PT_IDXn(x + 0, y + 1, z + 1));
            }
        }
    }

    attributeList.reserve(numCellsTotal);
    for (uint32_t cellIdx = 0; cellIdx < numCellsTotal; cellIdx++) {
        float cellAttribute;
        in >> cellAttribute;
        attributeList.push_back(cellAttribute);
    }

    std::vector<std::vector<float>> ret;
    ret.emplace_back(attributeList);


    return std::shared_ptr<SimVisData>(new SimVisData(numCellsX, numCellsY, numCellsZ, verts, cellInds, ret));
}

SimVisDataPtr SimVisData::loadSphere(int dim)
{
  SimVisDataPtr              simVisDataPtr = nullptr;
  std::vector<nvmath::vec3f> verts;
  std::vector<int>           cellInds;
  std::vector<float>         attributeList;

  
  

  uint32_t numCellsX = dim, numCellsY = dim, numCellsZ = dim;
 
  uint32_t numCellsTotal = numCellsX * numCellsY * numCellsZ;

  // add the vertices.
  for(uint32_t z = 0; z < numCellsZ + 1; z++)
  {
    for(uint32_t y = 0; y < numCellsY + 1; y++)
    {
      for(uint32_t x = 0; x < numCellsX + 1; x++)
      {
        verts.push_back(nvmath::vec3f(x, y, z));
      }
    }
  }

  // add the cell ids. Used if we deal with unstructure grid
  for(uint32_t z = 0; z < numCellsZ; z++)
  {
    for(uint32_t y = 0; y < numCellsY; y++)
    {
      for(uint32_t x = 0; x < numCellsX; x++)
      {
        cellInds.push_back(PT_IDXn(x + 0, y + 0, z + 0));
        cellInds.push_back(PT_IDXn(x + 1, y + 0, z + 0));
        cellInds.push_back(PT_IDXn(x + 1, y + 1, z + 0));
        cellInds.push_back(PT_IDXn(x + 0, y + 1, z + 0));
        cellInds.push_back(PT_IDXn(x + 0, y + 0, z + 1));
        cellInds.push_back(PT_IDXn(x + 1, y + 0, z + 1));
        cellInds.push_back(PT_IDXn(x + 1, y + 1, z + 1));
        cellInds.push_back(PT_IDXn(x + 0, y + 1, z + 1));
      }
    }
  }

  attributeList.reserve(numCellsTotal);
  for(uint32_t z = 0; z < numCellsZ; z++)
  {
    for(uint32_t y = 0; y < numCellsY; y++)
    {
      for(uint32_t x = 0; x < numCellsX; x++)
      {
        float d2 = ((x - dim / 2.f) * (x - dim / 2.f) + (y - dim / 2.f) * (y - dim / 2.f) + (z - dim / 2.f) * (z - dim / 2.f));
        float r2 = (dim * dim / 4.f);
        float cellAttribute =
            
                                    d2  < r2 ?
                                  (1 - std::sqrtf(d2 / r2)) :       // make attribute value between 0 and 1
                                  0.f;
        attributeList.push_back(cellAttribute);
      }
    }
  }

  std::vector<std::vector<float>> ret;
  ret.emplace_back(attributeList);


  return std::shared_ptr<SimVisData>(new SimVisData(numCellsX, numCellsY, numCellsZ, verts, cellInds, ret));
}