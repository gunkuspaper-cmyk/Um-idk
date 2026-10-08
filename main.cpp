// Geometrize Importer for Geometry Dash (Geode v5)
//
// Reads a Geometrize JSON export and rebuilds it in the level editor.
// Supported Geometrize shape types (the "type" field in the JSON):
//   1   rectangle            data: [x1, y1, x2, y2]
//   2   rotated rectangle    data: [x1, y1, x2, y2, angle]
//   4   triangle             data: [x1, y1, x2, y2, x3, y3]
//   8   ellipse              data: [x, y, rx, ry]
//   16  rotated ellipse      data: [x, y, rx, ry, angle]
//   32  circle               data: [x, y, r]
//   64  line                 data: [x1, y1, x2, y2]
//   128 quadratic bezier     data: [cx, cy, x1, y1, x2, y2]
//
// Colors are merged into a limited number of GD color channels with k-means.

#include <Geode/Geode.hpp>
#include <Geode/modify/EditorPauseLayer.hpp>
#include <Geode/utils/file.hpp>
#include <Geode/utils/async.hpp>
#include <matjson.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numbers>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace geode::prelude;

namespace {

constexpr float BLOCK = 30.f;                       // size of a 1x1 block in GD units
constexpr double RAD2DEG = 180.0 / std::numbers::pi;

enum ShapeType : int {
	kRectangle = 1,
	kRotatedRectangle = 2,
	kTriangle = 4,
	kEllipse = 8,
	kRotatedEllipse = 16,
	kCircle = 32,
	kLine = 64,
	kQuadraticBezier = 128,
};

struct Shape {
	int type = 0;
	std::vector<double> d;
	std::array<float, 4> rgba{ 0, 0, 0, 255 };
};

struct Config {
	int squareID, circleID, triangleID;
	bool rightAngleOnRight, importBackground;
	float unitsPerPixel, lineThickness;
	int bezierSegments, channelCount, startChannel, shapeLimit;

	static Config load() {
		auto m = Mod::get();
		Config c;
		c.squareID = static_cast<int>(m->getSettingValue<int64_t>("square-id"));
		c.circleID = static_cast<int>(m->getSettingValue<int64_t>("circle-id"));
		c.triangleID = static_cast<int>(m->getSettingValue<int64_t>("triangle-id"));
		c.rightAngleOnRight = m->getSettingValue<bool>("triangle-right-angle-right");
		c.importBackground = m->getSettingValue<bool>("import-background");
		c.unitsPerPixel = static_cast<float>(m->getSettingValue<double>("units-per-pixel"));
		c.lineThickness = static_cast<float>(m->getSettingValue<double>("line-thickness"));
		c.bezierSegments = static_cast<int>(m->getSettingValue<int64_t>("bezier-segments"));
		c.channelCount = static_cast<int>(m->getSettingValue<int64_t>("color-channels"));
		c.startChannel = static_cast<int>(m->getSettingValue<int64_t>("start-channel"));
		c.shapeLimit = static_cast<int>(m->getSettingValue<int64_t>("shape-limit"));
		// never run past channel 999
		c.channelCount = std::max(1, std::min(c.channelCount, 999 - c.startChannel + 1));
		return c;
	}
};

// ---------------------------------------------------------------- parsing

Result<std::vector<Shape>> parseShapes(std::filesystem::path const& path) {
	std::ifstream file(path);
	if (!file) {
		return Err(std::string("Could not open the file."));
	}
	auto parsed = matjson::parse(file);
	if (!parsed) {
		return Err(std::string("That file is not valid JSON."));
	}
	matjson::Value root = parsed.unwrap();

	matjson::Value const* list = &root;
	if (root.isObject() && root.contains("shapes")) {
		list = &root["shapes"];
	}
	if (!list->isArray()) {
		return Err(std::string("Expected a Geometrize JSON export (an array of shapes)."));
	}

	std::vector<Shape> shapes;
	for (auto const& el : *list) {
		Shape s;
		s.type = el["type"].asInt().unwrapOr(0);

		auto const& data = el["data"];
		if (data.isArray()) {
			for (auto const& v : data) {
				s.d.push_back(v.asDouble().unwrapOr(0.0));
			}
		}
		auto const& col = el["color"];
		if (col.isArray()) {
			for (size_t i = 0; i < 4; i++) {
				s.rgba[i] = static_cast<float>(col[i].asDouble().unwrapOr(i == 3 ? 255.0 : 0.0));
			}
		}
		shapes.push_back(std::move(s));
	}
	if (shapes.empty()) {
		return Err(std::string("The file contains no shapes."));
	}
	return Ok(std::move(shapes));
}

// Image size: Geometrize's first shape is a rectangle covering the whole image.
// Falls back to the extent of all coordinates if that isn't the case.
CCSize detectImageSize(std::vector<Shape> const& shapes) {
	auto const& f = shapes.front();
	if (f.type == kRectangle && f.d.size() >= 4) {
		float w = static_cast<float>(std::max(f.d[0], f.d[2]));
		float h = static_cast<float>(std::max(f.d[1], f.d[3]));
		if (w > 1 && h > 1) return { w, h };
	}
	double w = 1, h = 1;
	for (auto const& s : shapes) {
		size_t pairs = 0;
		switch (s.type) {
			case kEllipse: case kRotatedEllipse: case kCircle: pairs = 1; break;
			case kRotatedRectangle: pairs = 2; break;
			case kRectangle: case kLine: pairs = 2; break;
			case kTriangle: case kQuadraticBezier: pairs = 3; break;
			default: break;
		}
		for (size_t p = 0; p < pairs && p * 2 + 1 < s.d.size(); p++) {
			w = std::max(w, s.d[p * 2]);
			h = std::max(h, s.d[p * 2 + 1]);
		}
	}
	return { static_cast<float>(w), static_cast<float>(h) };
}

// ---------------------------------------------------------------- colors

struct Quantized {
	std::vector<std::array<float, 4>> centers;   // rgba, 0-255
	std::vector<int> assignment;                 // per shape -> index in centers
};

float dist2(std::array<float, 4> const& a, std::array<float, 4> const& b) {
	float s = 0;
	for (int i = 0; i < 4; i++) {
		float d = a[i] - b[i];
		s += d * d;
	}
	return s;
}

Quantized quantize(std::vector<Shape> const& shapes, int k) {
	Quantized q;
	q.assignment.assign(shapes.size(), 0);

	// exact distinct colors first
	std::unordered_map<uint32_t, int> distinct;
	auto key = [](std::array<float, 4> const& c) {
		auto b = [](float v) { return static_cast<uint32_t>(std::clamp(std::lround(v), 0L, 255L)); };
		return (b(c[0]) << 24) | (b(c[1]) << 16) | (b(c[2]) << 8) | b(c[3]);
	};
	for (size_t i = 0; i < shapes.size(); i++) {
		auto kk = key(shapes[i].rgba);
		auto it = distinct.find(kk);
		if (it == distinct.end()) {
			it = distinct.emplace(kk, static_cast<int>(q.centers.size())).first;
			q.centers.push_back(shapes[i].rgba);
		}
		q.assignment[i] = it->second;
	}
	if (static_cast<int>(q.centers.size()) <= k) {
		return q;
	}

	// k-means
	q.centers.clear();
	for (int i = 0; i < k; i++) {
		q.centers.push_back(shapes[(static_cast<size_t>(i) * shapes.size()) / k].rgba);
	}
	for (int iter = 0; iter < 10; iter++) {
		for (size_t i = 0; i < shapes.size(); i++) {
			float best = std::numeric_limits<float>::max();
			int bi = 0;
			for (int c = 0; c < k; c++) {
				float d = dist2(shapes[i].rgba, q.centers[c]);
				if (d < best) { best = d; bi = c; }
			}
			q.assignment[i] = bi;
		}
		std::vector<std::array<double, 5>> sum(k, { 0, 0, 0, 0, 0 });
		for (size_t i = 0; i < shapes.size(); i++) {
			auto& s = sum[q.assignment[i]];
			for (int j = 0; j < 4; j++) s[j] += shapes[i].rgba[j];
			s[4] += 1;
		}
		for (int c = 0; c < k; c++) {
			if (sum[c][4] > 0) {
				for (int j = 0; j < 4; j++) {
					q.centers[c][j] = static_cast<float>(sum[c][j] / sum[c][4]);
				}
			}
		}
	}
	return q;
}

// ---------------------------------------------------------------- placing objects

struct Placer {
	LevelEditorLayer* editor;
	Config cfg;
	CCPoint origin;       // editor position of the image center
	CCSize image;         // image size in Geometrize pixels
	int channel = 0;
	int zOrder = 0;
	size_t created = 0;

	CCPoint toGD(double x, double y) const {
		return {
			origin.x + static_cast<float>(x - image.width / 2.0) * cfg.unitsPerPixel,
			origin.y + static_cast<float>(image.height / 2.0 - y) * cfg.unitsPerPixel,
		};
	}
	float len(double pixels) const { return static_cast<float>(pixels) * cfg.unitsPerPixel; }

	// sizeX/sizeY are in GD units, rotCW in degrees clockwise (GD convention)
	void put(int id, CCPoint pos, float sizeX, float sizeY, float rotCW, bool flipY = false) {
		auto obj = editor->createObject(id, pos, false);
		if (!obj) return;
		obj->updateCustomScaleX(std::clamp(sizeX / BLOCK, 0.01f, 200.f));
		obj->updateCustomScaleY(std::clamp(sizeY / BLOCK, 0.01f, 200.f));
		obj->setRotation(rotCW);
		if (flipY) obj->setFlipY(true);
		obj->m_zOrder = zOrder;
		if (obj->m_baseColor) obj->m_baseColor->m_colorID = channel;
		if (obj->m_detailColor) obj->m_detailColor->m_colorID = channel;
		created++;
	}

	// thin rectangle from a to b (GD coordinates)
	void bar(CCPoint a, CCPoint b, float thickness) {
		float dx = b.x - a.x, dy = b.y - a.y;
		float length = std::hypot(dx, dy);
		if (length < 0.01f) return;
		put(cfg.squareID, { (a.x + b.x) / 2, (a.y + b.y) / 2 }, length, thickness,
			static_cast<float>(-std::atan2(dy, dx) * RAD2DEG));
	}

	// Right triangle with the right angle at F, legs along unit vectors dir1 / dir2
	// (GD coordinates, y up) with lengths l1 / l2.
	void rightTriangle(CCPoint F, CCPoint dir1, float l1, CCPoint dir2, float l2) {
		if (l1 < 0.01f || l2 < 0.01f) return;
		// Template: right angle on bottom-right (s = 1) or bottom-left (s = -1),
		// one leg along the bottom edge, the other along the vertical edge.
		float s = cfg.rightAngleOnRight ? 1.f : -1.f;
		float theta = std::atan2(-s * dir1.y, -s * dir1.x);          // CCW, math convention
		float px = -std::sin(theta), py = std::cos(theta);           // where the vertical leg points
		bool flip = (dir2.x * px + dir2.y * py) < 0;
		CCPoint center = {
			F.x + dir1.x * 0.5f * l1 + dir2.x * 0.5f * l2,
			F.y + dir1.y * 0.5f * l1 + dir2.y * 0.5f * l2,
		};
		put(cfg.triangleID, center, l1, l2, static_cast<float>(-theta * RAD2DEG), flip);
	}

	// Any triangle = two right triangles split by the altitude onto the longest edge.
	void triangle(CCPoint a, CCPoint b, CCPoint c) {
		std::array<CCPoint, 3> p{ a, b, c };
		auto d2 = [&](int i, int j) {
			float dx = p[i].x - p[j].x, dy = p[i].y - p[j].y;
			return dx * dx + dy * dy;
		};
		int pi = 0, qi = 1, ri = 2;
		float best = d2(0, 1);
		if (d2(1, 2) > best) { best = d2(1, 2); pi = 1; qi = 2; ri = 0; }
		if (d2(2, 0) > best) { best = d2(2, 0); pi = 2; qi = 0; ri = 1; }
		if (best < 1e-4f) return;

		CCPoint P = p[pi], Q = p[qi], R = p[ri];
		float pqx = Q.x - P.x, pqy = Q.y - P.y;
		float t = ((R.x - P.x) * pqx + (R.y - P.y) * pqy) / best;
		CCPoint F = { P.x + pqx * t, P.y + pqy * t };

		float hx = R.x - F.x, hy = R.y - F.y;
		float h = std::hypot(hx, hy);
		if (h < 0.01f) return;
		CCPoint dirH = { hx / h, hy / h };

		for (CCPoint end : { P, Q }) {
			float ex = end.x - F.x, ey = end.y - F.y;
			float l = std::hypot(ex, ey);
			if (l < 0.01f) continue;
			rightTriangle(F, { ex / l, ey / l }, l, dirH, h);
		}
	}

	void place(Shape const& s) {
		auto const& d = s.d;
		auto need = [&](size_t n) { return d.size() >= n; };
		switch (s.type) {
			case kRectangle: {
				if (!need(4)) return;
				auto c = toGD((d[0] + d[2]) / 2, (d[1] + d[3]) / 2);
				put(cfg.squareID, c, len(std::abs(d[2] - d[0])), len(std::abs(d[3] - d[1])), 0);
			} break;
			case kRotatedRectangle: {
				if (!need(5)) return;
				auto c = toGD((d[0] + d[2]) / 2, (d[1] + d[3]) / 2);
				// Geometrize angles are clockwise on screen, same as GD rotation
				put(cfg.squareID, c, len(std::abs(d[2] - d[0])), len(std::abs(d[3] - d[1])),
					static_cast<float>(d[4]));
			} break;
			case kTriangle: {
				if (!need(6)) return;
				triangle(toGD(d[0], d[1]), toGD(d[2], d[3]), toGD(d[4], d[5]));
			} break;
			case kEllipse: {
				if (!need(4)) return;
				put(cfg.circleID, toGD(d[0], d[1]), len(d[2] * 2), len(d[3] * 2), 0);
			} break;
			case kRotatedEllipse: {
				if (!need(5)) return;
				put(cfg.circleID, toGD(d[0], d[1]), len(d[2] * 2), len(d[3] * 2),
					static_cast<float>(d[4]));
			} break;
			case kCircle: {
				if (!need(3)) return;
				put(cfg.circleID, toGD(d[0], d[1]), len(d[2] * 2), len(d[2] * 2), 0);
			} break;
			case kLine: {
				if (!need(4)) return;
				bar(toGD(d[0], d[1]), toGD(d[2], d[3]), len(cfg.lineThickness));
			} break;
			case kQuadraticBezier: {
				if (!need(6)) return;
				// Geometrize order: control point, start, end
				double cx = d[0], cy = d[1], x1 = d[2], y1 = d[3], x2 = d[4], y2 = d[5];
				int n = cfg.bezierSegments;
				CCPoint prev = toGD(x1, y1);
				for (int i = 1; i <= n; i++) {
					double t = static_cast<double>(i) / n, u = 1.0 - t;
					double x = u * u * x1 + 2 * u * t * cx + t * t * x2;
					double y = u * u * y1 + 2 * u * t * cy + t * t * y2;
					CCPoint cur = toGD(x, y);
					bar(prev, cur, len(cfg.lineThickness));
					prev = cur;
				}
			} break;
			default:
				break; // unknown shape type, skipped
		}
	}
};

void showError(std::string const& msg) {
	FLAlertLayer::create("Geometrize Importer", msg, "OK")->show();
}

void importFile(std::filesystem::path const& path) {
	auto editor = LevelEditorLayer::get();
	if (!editor) return;

	auto parsed = parseShapes(path);
	if (parsed.isErr()) {
		showError(parsed.unwrapErr());
		return;
	}
	auto shapes = parsed.unwrap();

	auto cfg = Config::load();
	auto image = detectImageSize(shapes);

	size_t first = cfg.importBackground ? 0 : 1;
	size_t last = shapes.size();
	if (cfg.shapeLimit > 0) {
		last = std::min(last, first + static_cast<size_t>(cfg.shapeLimit));
	}
	std::vector<Shape> used(shapes.begin() + first, shapes.begin() + last);
	if (used.empty()) {
		showError("There are no shapes left to import with the current settings.");
		return;
	}

	// color channels
	auto q = quantize(used, cfg.channelCount);
	auto fx = editor->m_effectManager;
	for (size_t i = 0; i < q.centers.size(); i++) {
		auto action = fx->getColorAction(cfg.startChannel + static_cast<int>(i));
		if (!action) continue;
		auto const& c = q.centers[i];
		action->m_color = {
			static_cast<GLubyte>(std::clamp(std::lround(c[0]), 0L, 255L)),
			static_cast<GLubyte>(std::clamp(std::lround(c[1]), 0L, 255L)),
			static_cast<GLubyte>(std::clamp(std::lround(c[2]), 0L, 255L)),
		};
		action->m_opacity = std::clamp(c[3] / 255.f, 0.f, 1.f);
	}

	// where to put it: the middle of what the editor is currently showing
	auto win = CCDirector::get()->getWinSize();
	auto layer = editor->m_objectLayer;
	CCPoint center = {
		(win.width / 2 - layer->getPositionX()) / layer->getScale(),
		(win.height / 2 - layer->getPositionY()) / layer->getScale(),
	};

	Placer placer{ editor, cfg, center, image };
	for (size_t i = 0; i < used.size(); i++) {
		placer.channel = cfg.startChannel + q.assignment[i];
		placer.zOrder = static_cast<int>(i);
		placer.place(used[i]);
	}

	Notification::create(
		fmt::format("Geometrize: created {} objects", placer.created),
		NotificationIcon::Success
	)->show();
}

} // namespace

class $modify(GeometrizeEditorPause, EditorPauseLayer) {
	bool init(LevelEditorLayer* editor) {
		if (!EditorPauseLayer::init(editor)) return false;

		auto sprite = ButtonSprite::create("Geometrize", "goldFont.fnt", "GJ_button_04.png", .7f);
		auto button = CCMenuItemSpriteExtra::create(
			sprite, this, menu_selector(GeometrizeEditorPause::onGeometrize)
		);
		button->setID("geometrize-import-button"_spr);

		auto menu = CCMenu::create();
		menu->setID("geometrize-menu"_spr);
		menu->addChild(button);
		auto win = CCDirector::get()->getWinSize();
		menu->setPosition({ 70.f, win.height - 30.f });
		this->addChild(menu);
		return true;
	}

	void onGeometrize(CCObject*) {
		file::FilePickOptions options;
		options.filters.push_back({ "Geometrize JSON", { "*.json" } });

		async::spawn(
			file::pick(file::PickMode::OpenFile, options),
			[](Result<std::optional<std::filesystem::path>> result) {
				if (result.isErr()) {
					showError("Could not open the file picker.");
					return;
				}
				auto path = result.unwrap();
				if (!path) return; // cancelled
				importFile(*path);
			}
		);
	}
};
