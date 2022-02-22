#include <memory>
#include <vector>
#include <nvmath/nvmath.h>

class SimVisData;
typedef std::shared_ptr<SimVisData> SimVisDataPtr;
#define PT_IDXn(x, y, z) ((x) + (y) * (numCellsX + 1) + (z) * (numCellsY + 1) * (numCellsX + 1))


class SimVisData
{
public:
	SimVisData() = delete;

	static SimVisDataPtr loadFromFile(const char* filepath);

    static SimVisDataPtr loadSphere(int dim);

    std::vector<int> cellIndices;

    std::vector<nvmath::vec3f> vertices;
	
    std::vector<std::vector<float>> attributesList;
    
    uint32_t                        numCellsX, numCellsY, numCellsZ;

  private:
    SimVisData(uint32_t numX, uint32_t numY, uint32_t numZ,
        const std::vector<nvmath::vec3f>& verts,
        const std::vector<int>&                inds,
        const std::vector<std::vector<float>>& attributesList)
        : vertices(verts)
        , cellIndices(inds)
        , attributesList(attributesList)
        , numCellsX(numX)
        , numCellsY(numY)
        , numCellsZ(numZ)
		{}

};