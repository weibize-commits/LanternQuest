import { afterEach, describe, expect, it, vi } from "vitest";

import { generatePattern, PatternClientError } from "./patternClient";

afterEach(() => vi.unstubAllGlobals());

describe("pattern client", () => {
  it("accepts an image returned by the same-origin proxy", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      asset_id: "pattern-1",
      image_data_url: "data:image/webp;base64,UklGRg==",
      model: "gpt-image-test",
      created_at: "2026-09-25T00:00:00Z",
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    const result = await generatePattern("桂花与流水");
    expect(result.asset_id).toBe("pattern-1");
    expect(fetchMock).toHaveBeenCalledWith("/api/pattern", expect.objectContaining({ method: "POST" }));
  });

  it("surfaces a proxy error without fabricating a fallback image", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      code: "pattern_api_unconfigured",
      message: "图案生成尚未配置",
    }), { status: 503, headers: { "Content-Type": "application/json" } })));
    await expect(generatePattern("测试")).rejects.toMatchObject<Partial<PatternClientError>>({
      code: "pattern_api_unconfigured",
    });
  });
});
